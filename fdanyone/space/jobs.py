"""Local UI jobs delegated to the validated multi-GPU queue."""
from __future__ import annotations

import json
import math
from fractions import Fraction
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
import uuid

from fdanyone.io import write_json
from fdanyone.views import resolve_view_plan

ROOT = Path(__file__).resolve().parents[2]
JOB_ID = re.compile(r'job-[0-9a-f]{12}')


def _read_json(path, default=None):
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {} if default is None else default


def _tree_bytes(directory):
    """Count owned files without following links outside a job."""
    if directory.is_symlink():
        return 0
    total = 0
    for root, _, files in os.walk(directory, followlinks=False):
        for name in files:
            path = Path(root)/name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def has_generated_result(job):
    """Keep successfully generated views accessible if downstream training fails."""
    if job.get('status') == 'completed':
        return True
    if job.get('status') not in ('failed', 'cancelled') or not job.get('result_dir'):
        return False
    from fdanyone.reconstruction import training_status
    return ((Path(job['result_dir'])/'metadata.json').is_file()
            and training_status(job['result_dir']).get('status') in ('failed', 'cancelled'))


def make_options(views, pitches, yaw, span, turbo, start_time, fps, seed):
    values = (views, yaw, span, seed)
    if any(isinstance(v, bool) or int(v) != v for v in values):
        raise ValueError('Camera counts, yaw, span and seed must be integers.')
    pitches = json.loads(pitches) if isinstance(pitches, str) else pitches
    plan = resolve_view_plan(views_per_layer=int(views), layer_pitches=pitches,
                             start_yaw=int(yaw), yaw_span=int(span), enable_rcp=not turbo)
    if turbo and (plan.num_layers > 3 or plan.num_target_views not in (6, 8, 12, 16, 18, 24, 36)):
        raise ValueError('Turbo supports 6, 8, 12, 16, 18, 24 or 36 cameras across up to three rings.')
    if int(seed) < 0:
        raise ValueError('Seed must be non-negative.')
    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError('Start time must be finite and non-negative.')
    if str(fps).lower() != 'auto' and Fraction(str(fps)) <= 0:
        raise ValueError('FPS must be positive.')
    return dict(views_per_layer=plan.views_per_layer, layer_pitches=list(plan.layer_pitches),
                start_yaw=plan.start_yaw, yaw_span=plan.yaw_span, turbo=bool(turbo),
                start_time=float(start_time), target_fps=fps, seed=int(seed),
                execution_profile='full', compile_dit=True, motion_backend='auto', motion_precision='fp16')


class JobManager:
    def __init__(self, cache_dir, model_dir='models'):
        self.root = Path(cache_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self.process = None
        reports = list(self.root.glob('job-*/results/streams_report.json'))
        self.directory = max(reports, key=lambda p:p.stat().st_mtime_ns).parent.parent if reports else None
        self.lock = threading.RLock()
        from .workflow import WorkflowStore
        self.workflow = WorkflowStore(self)
        workflows = list(self.root.glob('job-*/job.json'))
        if workflows:
            active = [p for p in workflows if _read_json(p).get('active')]
            self.directory = max(active or workflows, key=lambda p:p.stat().st_mtime_ns).parent

    def start(self, videos, options, *, clip_plan=None, source_names=None):
        with self.lock:
            if self.workflow.busy():
                raise ValueError('A task is already running. Wait for it or cancel it first.')
            paths = [str(Path(p).expanduser().resolve()) for p in videos or []]
            if not paths or any(not Path(p).is_file() for p in paths):
                raise ValueError('Choose at least one existing input video.')
            if options.get('train_4dgs'):
                from fdanyone.reconstruction import check_runtime
                check_runtime(probe=True)
            directory = self.root / ('job-' + uuid.uuid4().hex[:12])
            directory.mkdir()
            request = dict(video_paths=paths, output_dir=str(directory/'results'),
                           model_dir=self.model_dir, **options)
            if clip_plan is not None:
                request['_clip_plan'] = clip_plan
            write_json(directory/'request.json', request)
            write_json(directory/'ui.json', {'source_names': source_names or [Path(p).name for p in paths]})
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(ROOT)
            with (directory/'launcher.log').open('wb') as log:
                self.process = subprocess.Popen(
                    [sys.executable, '-m', 'fdanyone.space.jobs', str(directory/'request.json')],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            self.directory = directory
            return str(directory)

    def job_directory(self, job_id):
        if not JOB_ID.fullmatch(job_id):
            raise ValueError('Invalid job ID.')
        directory = self.root/job_id
        if directory.is_symlink() or directory.resolve() != directory or not directory.is_dir():
            raise ValueError('This job is no longer available.')
        return directory

    def preview_root(self, directory):
        """Keep every managed job's playback copies in its own namespace."""
        directory = Path(directory).expanduser().resolve()
        if directory.is_relative_to(self.root):
            parts = directory.relative_to(self.root).parts
            if parts and JOB_ID.fullmatch(parts[0]):
                self.job_directory(parts[0])
                return self.root/'previews'/parts[0]
        return self.root/'previews'/'external'

    def _completed(self, directory):
        if (directory/'job.json').is_file():
            from .workflow import selected_run, stale
            job = self.workflow.load(directory.name)
            generated = selected_run(job, 'generate')
            training = selected_run(job, 'splat')
            results = []
            if generated:
                for item in generated['artifacts']:
                    result = Path(item['result_dir']).resolve()
                    if result.is_relative_to(directory) and result.is_dir():
                        matching = next((a for a in (training or {}).get('artifacts', []) if a['result_dir'] == str(result)), None)
                        results.append(dict(directory=str(result), label=item['label'],
                                            training='completed' if matching else None))
            config = generated['config'] if generated else job['configs']['generate']
            return dict(id=job['id'], directory=str(directory), name=job['source']['name'], workflow=True,
                        state='running' if job['active'] else ('completed' if results else 'draft'),
                        completed_at=job['updated_at'], cameras=config['views']*len(config['pitches']),
                        chunks=len(results), results=results,
                        bytes=_tree_bytes(directory)+_tree_bytes(self.root/'previews'/directory.name))
        if directory == self.directory and self.process is not None and self.process.poll() is None:
            return None
        report_path = directory/'results/streams_report.json'
        report = _read_json(report_path)
        jobs = report.get('jobs', [])
        if not isinstance(jobs, list) or not jobs or any(not isinstance(j, dict) or not has_generated_result(j) for j in jobs):
            return None
        request = _read_json(directory/'request.json')
        ui = _read_json(directory/'ui.json')
        names = ui.get('source_names', [])
        if not names:
            names = [Path(p).name for p in request.get('video_paths', [])]
        name = names[0] if names else 'Video'
        if re.fullmatch(r'[0-9a-f]{32}\.[a-zA-Z0-9]+', name):
            name = 'Video'
        if len(names) > 1:
            name += f' + {len(names)-1} more'
        chunks = request.get('_clip_plan', {}).get('chunks', [])
        results = []
        for index, job in enumerate(jobs):
            result = Path(job.get('result_dir', '')).resolve()
            if not result.is_relative_to(directory) or not result.is_dir():
                continue
            from fdanyone.reconstruction import training_status
            results.append({'directory': str(result), 'label': chunks[index]['label'] if index < len(chunks) else f'Video {index+1}',
                            'training': training_status(result).get('status')})
        return dict(id=directory.name, directory=str(directory), name=name,
                    completed_at=report_path.stat().st_mtime,
                    cameras=request.get('views_per_layer', 0)*len(request.get('layer_pitches', [])),
                    chunks=len(jobs), results=results,
                    bytes=_tree_bytes(directory)+_tree_bytes(self.root/'previews'/directory.name))

    def completed(self):
        with self.lock:
            records = []
            for path in self.root.glob('job-*'):
                try:
                    directory = self.job_directory(path.name)
                    record = self._completed(directory)
                    if record:
                        records.append(record)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            return sorted(records, key=lambda row: row['completed_at'], reverse=True)

    def completed_job(self, job_id):
        with self.lock:
            return self._completed(self.job_directory(job_id))

    def training_results(self, job_id, run_id=None):
        """Owned results with training state, including active/partial models."""
        from fdanyone.reconstruction import training_status
        with self.lock:
            directory = self.job_directory(job_id)
            if (directory/'job.json').is_file():
                from .workflow import selected_run
                job = self.workflow.load(job_id)
                if run_id:
                    run = next((r for r in job['runs'] if r['id'] == run_id and r['stage'] == 'splat'), None)
                else:
                    # Include live training; URLs pin its immutable attempt so a
                    # later selection cannot change an in-flight download.
                    run = next((r for r in reversed(job['runs']) if r['stage'] == 'splat' and r['status'] == 'running'), None) or selected_run(job, 'splat')
                results = []
                for item in (run or {}).get('artifacts', []):
                    output = Path(item['training_dir']).resolve()
                    if output.is_relative_to(directory):
                        status = _read_json(output/'status.json')
                        if status:
                            results.append(dict(chunk=item['chunk'], directory=item['result_dir'],
                                                output_dir=str(output), training=status, run_id=run['id']))
                return results
            jobs = _read_json(directory/'results/streams_report.json').get('jobs', [])
            results = []
            for index, job in enumerate(jobs):
                if not isinstance(job, dict) or not job.get('result_dir'):
                    continue
                result = Path(job['result_dir']).resolve()
                if result.is_relative_to(directory) and result.is_dir():
                    status = training_status(result)
                    if status:
                        results.append({'chunk': index, 'directory': str(result), 'training': status})
            return results

    def delete_completed(self, job_id):
        with self.lock:
            directory = self.job_directory(job_id)
            if ((directory/'job.json').is_file() and self.workflow.load(job_id).get('active')) or not self._completed(directory):
                raise ValueError('Only completed jobs can be deleted. Running and unfinished jobs are protected.')
            previews = self.root/'previews'/job_id
            if previews.is_symlink() or previews.resolve() != previews:
                raise ValueError('Invalid job preview directory.')
            # Never follow report paths for deletion. Only these two owned trees
            # can be removed; uploads, external results and model assets stay put.
            if previews.exists():
                shutil.rmtree(previews)
            shutil.rmtree(directory)
            if self.directory == directory:
                reports = list(self.root.glob('job-*/results/streams_report.json'))
                self.directory = max(reports, key=lambda p:p.stat().st_mtime_ns).parent.parent if reports else None
                self.process = None
            return {'deleted': job_id}

    def snapshot(self):
        with self.lock:
            if self.directory is None:
                return {'status': 'idle', 'jobs': []}, ''
            if (self.directory/'job.json').is_file():
                job = self.workflow.load(self.directory.name)
                active = job.get('active')
                latest = next((r for r in job['runs'] if r['status'] == 'running'), None) or (job['runs'][-1] if job['runs'] else None)
                data = dict(jobs=[], status='running' if active else (latest['status'] if latest else 'idle'),
                            directory=str(self.directory), job_id=job['id'], workflow=True,
                            stage=latest['stage'] if latest else 'trim', active=active)
                logs = []
                if latest:
                    stage_root = self.directory/'stages'/latest['id']
                    data.update(_read_json(stage_root/'streams_report.json'))
                    for i, row in enumerate(data['jobs']):
                        artifacts = latest.get('artifacts', [])
                        if i < len(artifacts) and artifacts[i].get('training_dir'):
                            row['training'] = _read_json(Path(artifacts[i]['training_dir'])/'status.json')
                    data['run_id'] = latest['id']
                    data['error'] = latest.get('error')
                    logs = [Path(j['log']) for j in data['jobs']]
                chunks = []
                for path in [self.directory/'launcher.log', *logs]:
                    if path.is_file():
                        with path.open('rb') as stream:
                            stream.seek(max(0, path.stat().st_size-6000))
                            chunks.append(path.name+'\n'+stream.read().decode(errors='replace'))
                return data, '\n\n'.join(chunks)[-20000:]
            report = self.directory/'results/streams_report.json'
            data = json.loads(report.read_text()) if report.exists() else {'jobs': []}
            code = self.process.poll() if self.process else (0 if data['jobs'] and all(j['status'] == 'completed' for j in data['jobs']) else 1)
            data['status'] = 'running' if code is None else ('completed' if code == 0 else 'stopped or failed')
            data['directory'] = str(self.directory)
            request_path = self.directory/'request.json'
            request = json.loads(request_path.read_text()) if request_path.exists() else {}
            plan = request.get('_clip_plan')
            if plan:
                data['stage'] = 'preparing clips' if not report.exists() and code is None else 'generation'
                data['planned_chunks'] = len(plan['chunks'])
                for job, chunk in zip(data['jobs'], plan['chunks']):
                    job['label'] = chunk['label']
                    job['source_start'] = chunk['start']
                    job['source_end'] = chunk['end']
            from fdanyone.reconstruction import training_status
            for job in data['jobs']:
                training = training_status(job.get('result_dir', ''))
                if training:
                    job['training'] = training
            active = [j['training'] for j in data['jobs'] if j.get('status') == 'running' and j.get('training')]
            if active:
                stages = {t.get('stage') for t in active}
                data['stage'] = next((s for s in ('training', 'preparing training views', 'packaging 4D scene')
                                      if s in stages), 'training')
            paths = [self.directory/'launcher.log'] + [Path(j['log']) for j in data['jobs']]
            chunks = []
            for path in paths:
                if path.exists():
                    with path.open('rb') as stream:
                        stream.seek(max(0, path.stat().st_size-6000))
                        chunks.append(path.name + '\n' + stream.read().decode(errors='replace'))
            return data, '\n\n'.join(chunks)[-20000:]

    def cancel(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                # The child turns SIGTERM into KeyboardInterrupt; run_queue's
                # finally block terminates its separate GPU process groups.
                self.process.wait(timeout=30)
            elif self.directory and (self.directory/'job.json').is_file():
                job = self.workflow.load(self.directory.name)
                active = job.get('active')
                if active:
                    pid = active.get('pid')
                    try:
                        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
                    except (ValueError, TypeError, OSError):
                        raise ValueError('The task process is unavailable. Refresh its status.') from None
                    if b'fdanyone.space.workflow' not in command or str(self.directory/'job.json').encode() not in command:
                        raise ValueError('The task process could not be verified.')
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic()+30
                    while self.workflow.busy() and time.monotonic()<deadline:
                        time.sleep(.1)
                    if self.workflow.busy():
                        raise ValueError('Stop requested; the task is still shutting down.')
            return 'Stopped. Completed outputs have been preserved.'

    def close(self):
        # A second viewer/server must not cancel another server's owned queue.
        if self.process is not None and self.process.poll() is None:
            self.cancel()


def main():
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    from fdanyone.streams import batch_inference
    try:
        request_path = Path(sys.argv[1])
        request = json.loads(request_path.read_text())
        plan = request.pop('_clip_plan', None)
        if plan:
            from fdanyone.space.clips import prepare_chunks
            print(f"Preparing {len(plan['chunks'])} lossless chunks…", flush=True)
            request['video_paths'] = prepare_chunks(plan, request_path.parent/'clips')
            request['start_time'] = 0
            request['target_fps'] = f"{plan['fps_num']}/{plan['fps_den']}"
        batch_inference(**request)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == '__main__':
    main()
