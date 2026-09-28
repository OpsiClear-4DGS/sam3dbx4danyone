"""Persistent jobs with immutable stage attempts and explicit artifact lineage.

The HTTP process edits drafts; a single queue process owns running jobs. A file
lock also protects against overlapping launches from separate server processes.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from fdanyone.io import write_json

STAGES = ('trim', 'pose', 'generate', 'splat')
LABELS = dict(trim='Trim', pose='Extract pose', generate='Generate videos', splat='Splat 4DGS')
DEFAULTS = dict(trim=dict(start=0., end=None, skip=0), pose=dict(precision='fp16'),
                generate=dict(views=6, pitches=[15], yaw=0, span=360, turbo=True, seed=42),
                splat=dict(steps=30000, samples_per_keyframe=32768))


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def owned(directory, path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError('Artifact is outside this job.')
    return path


def selected_run(job, stage):
    return next((r for r in job['runs'] if r['id'] == job['selected'].get(stage)), None)


def stale(job, run):
    return any(job['selected'].get(stage) != run_id for stage, run_id in run.get('inputs', {}).items())


def config_for(stage, values):
    if stage not in STAGES or not isinstance(values, dict):
        raise ValueError('Unknown pipeline stage or invalid settings.')
    if set(values) - set(DEFAULTS[stage]):
        raise ValueError(f'Unknown {stage} settings: {sorted(set(values) - set(DEFAULTS[stage]))}')
    config = dict(DEFAULTS[stage], **values)
    if stage == 'trim':
        if any(isinstance(config[k], bool) or not isinstance(config[k], (int, float)) or not math.isfinite(config[k])
               for k in ('start', 'end') if config[k] is not None):
            raise ValueError('Trim times must be finite numbers.')
        if config['start'] is None or config['start'] < 0 or (config['end'] is not None and config['end'] <= config['start']):
            raise ValueError('Choose an end after the start.')
        if type(config['skip']) is not int or not 0 <= config['skip'] <= 3:
            raise ValueError('Video speed must be between 1× and 4×.')
    elif stage == 'pose':
        if config['precision'] not in ('fp16', 'fp32'):
            raise ValueError('Choose fp16 or fp32 pose precision.')
    elif stage == 'generate':
        from .jobs import make_options
        if type(config['turbo']) is not bool:
            raise ValueError('Choose Turbo or Base generation.')
        if any(type(config[k]) is not int for k in ('views','yaw','span','seed')) or not isinstance(config['pitches'], list):
            raise ValueError('Camera counts, angles and seed must be integers.')
        make_options(config['views'], config['pitches'], config['yaw'], config['span'], config['turbo'], 0., 'auto', config['seed'])
    elif stage == 'splat':
        if type(config['steps']) is not int or not 1 <= config['steps'] <= 100000:
            raise ValueError('Training steps must be between 1 and 100,000.')
        if type(config['samples_per_keyframe']) is not int or not 4 <= config['samples_per_keyframe'] <= 131072:
            raise ValueError('Samples per keyframe must be between 4 and 131,072.')
    return config


class WorkflowStore:
    def __init__(self, manager):
        self.manager = manager
        self.root = manager.root

    @contextmanager
    def editing(self):
        # Short metadata mutations serialize across server threads/processes.
        with (self.root/'.jobs.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def busy(self):
        if self.manager.process is not None and self.manager.process.poll() is None:
            return True
        with (self.root/'.pipeline.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False

    def save(self, job):
        job['updated_at'] = time.time()
        write_json(self.root/job['id']/'job.json', job)

    def create(self, source):
        with self.editing():
            job_id = 'job-' + uuid.uuid4().hex[:12]
            (self.root/job_id).mkdir()
            job = dict(version=1, id=job_id, source=source, created_at=time.time(), updated_at=time.time(),
                       configs=json.loads(json.dumps(DEFAULTS)), selected={}, runs=[], active=None)
            self.save(job)
            return job

    def load(self, job_id):
        directory = self.manager.job_directory(job_id)
        with self.editing():
            job = read(directory/'job.json') or self.adopt(directory)
            if job.get('active') and not self.busy():
                # A process or machine restart preserves completed artifacts, but
                # never presents an abandoned attempt as still running.
                for run in job['runs']:
                    if run['status'] in ('running', 'queued'):
                        run.update(status='interrupted', error='Processing was interrupted. Run this step again.')
                job['active'] = None
                self.save(job)
        return job

    def describe(self, job_id):
        job = self.load(job_id)
        job['directory'] = str(self.root/job_id)
        job['busy'] = self.busy()
        for run in job['runs']:
            run['stale'] = stale(job, run)
        ready = {}
        for stage in STAGES:
            try:
                self.require_inputs(job, stage)
                ready[stage] = None
            except ValueError as exc:
                ready[stage] = str(exc)
        job['blocked'] = ready
        return job

    def adopt(self, directory):
        """Open existing gallery results as jobs without moving their files."""
        request = read(directory/'request.json')
        rows = read(directory/'results/streams_report.json').get('jobs', [])
        if not request or not rows:
            raise ValueError('This job has no saved workflow or reusable results.')
        plan = request.get('_clip_plan', {})
        paths = request.get('video_paths', [])
        source = paths[0] if paths else ''
        name = (read(directory/'ui.json').get('source_names') or [Path(source).name or 'Video'])[0]
        token = Path(source).name if Path(source).parent == self.root/'uploads' else None
        job = dict(version=1, id=directory.name, source=dict(path=source, name=name, id=token,
                   url='/media/uploads/'+token if token else None), created_at=directory.stat().st_mtime,
                   configs=json.loads(json.dumps(DEFAULTS)), selected={}, runs=[], active=None)
        job['configs']['trim'] = {k:plan.get(k, v) for k,v in DEFAULTS['trim'].items()}
        job['configs']['generate'] = dict(views=request.get('views_per_layer', 6), pitches=request.get('layer_pitches', [15]),
            yaw=request.get('start_yaw', 0), span=request.get('yaw_span', 360), turbo=request.get('turbo', True), seed=request.get('seed', 42))
        artifacts = {s:[] for s in STAGES}
        for i, row in enumerate(rows):
            result = owned(directory, row.get('result_dir', directory))
            if not (result/'metadata.json').is_file():
                continue
            worker = read(Path(row.get('request', directory/'missing')))
            video = worker.get('video_path', str(directory/'clips'/f'chunk_{i+1:03d}.mp4'))
            fps = read(result/'metadata.json').get('output', {}).get('fps', '25/1')
            label = plan.get('chunks', [])[i]['label'] if i < len(plan.get('chunks', [])) else f'Chunk {i+1}'
            item = dict(chunk=i, label=label, video=video, fps=fps, start_time=worker.get('start_time',0))
            artifacts['trim'].append(item)
            artifacts['pose'].append(dict(item, pose_dir=str(result/'preprocessing')))
            artifacts['generate'].append(dict(item, result_dir=str(result)))
            if (result/'training/status.json').is_file():
                artifacts['splat'].append(dict(item, result_dir=str(result), training_dir=str(result/'training')))
        for stage in STAGES:
            if not artifacts[stage]:
                continue
            if stage == 'splat' and any(read(Path(a['training_dir'])/'status.json').get('status') != 'completed' for a in artifacts[stage]):
                continue
            run = dict(id=stage+'-imported', stage=stage, status='completed', config=job['configs'][stage],
                       inputs=dict(job['selected']), artifacts=artifacts[stage], created_at=job['created_at'], imported=True)
            job['runs'].append(run)
            job['selected'][stage] = run['id']
        self.save(job)
        return job

    def require_inputs(self, job, stage):
        if stage == 'trim':
            if not Path(job['source']['path']).is_file():
                raise ValueError('The uploaded source is unavailable.')
            return {}
        if stage not in STAGES:
            raise ValueError('Unknown pipeline stage.')
        previous = STAGES[STAGES.index(stage)-1]
        run = selected_run(job, previous)
        if not run or run['status'] != 'completed':
            raise ValueError(f'Run {LABELS[previous]} first, or select a completed version.')
        if stale(job, run):
            raise ValueError(f'{LABELS[previous]} uses earlier inputs. Select that version or rerun it.')
        if not run.get('artifacts'):
            raise ValueError(f'{LABELS[previous]} has no saved output.')
        for artifact in run['artifacts']:
            if stage in ('pose', 'generate') and not Path(artifact['video']).is_file():
                raise ValueError('Prepared clips are unavailable. Run Trim again.')
            if stage == 'generate' and not (owned(self.root/job['id'], artifact['pose_dir'])/'sam3d_predictions.npz').is_file():
                raise ValueError('Saved pose is unavailable. Run Extract pose again.')
            if stage == 'splat' and not (owned(self.root/job['id'], artifact['result_dir'])/'metadata.json').is_file():
                raise ValueError('Generated videos are unavailable. Run Generate videos again.')
        return dict(run.get('inputs', {}), **{previous:run['id']})

    def configure(self, job_id, configs):
        normalized = {stage:config_for(stage, values) for stage,values in configs.items()}
        with self.editing():
            if self.busy():
                raise ValueError('Wait for the active task to finish before changing job settings.')
            job = read(self.manager.job_directory(job_id)/'job.json')
            if not job:
                raise ValueError('Open this job first.')
            job['configs'].update(normalized)
            self.save(job)
        return self.describe(job_id)

    def select(self, job_id, run_id):
        with self.editing():
            if self.busy():
                raise ValueError('Wait for the active task to finish before changing versions.')
            job = read(self.manager.job_directory(job_id)/'job.json')
            run = next((r for r in job.get('runs', []) if r['id'] == run_id), None)
            if not run or run['status'] != 'completed':
                raise ValueError('Select a completed version.')
            job['selected'].update(run['inputs'])
            job['selected'][run['stage']] = run['id']
            for stage, selected in job['selected'].items():
                if stage in run['inputs'] or stage == run['stage']:
                    chosen = next(r for r in job['runs'] if r['id'] == selected)
                    job['configs'][stage] = chosen['config']
            self.save(job)
        return self.describe(job_id)

    def start(self, job_id, start, through, configs):
        if start not in STAGES or through not in STAGES or STAGES.index(through) < STAGES.index(start):
            raise ValueError('Choose an end step at or after the start step.')
        normalized = {stage:config_for(stage, values) for stage,values in configs.items()}
        with self.editing():
            if self.busy():
                raise ValueError('A task is already running. Wait or stop it first.')
            directory = self.manager.job_directory(job_id)
            job = read(directory/'job.json')
            if not job:
                raise ValueError('Open this job first.')
            inputs = self.require_inputs(job, start)
            job['configs'].update(normalized)
            # Validate trim before committing any attempts or occupying a GPU.
            if start == 'trim':
                from .clips import plan_clip
                plan_clip(job['source']['path'], **job['configs']['trim'])
            stages = STAGES[STAGES.index(start):STAGES.index(through)+1]
            if 'splat' in stages:
                from fdanyone.reconstruction import check_runtime
                check_runtime()
            attempts = []
            for stage in stages:
                run_id = stage+'-'+uuid.uuid4().hex[:12]
                run = dict(id=run_id, stage=stage, status='queued', config=job['configs'][stage],
                           inputs=dict(inputs), artifacts=[], created_at=time.time())
                job['runs'].append(run)
                attempts.append(run_id)
                inputs[stage] = run_id
            lock = (self.root/'.pipeline.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                job['active'] = dict(runs=attempts, start=start, through=through)
                self.save(job)
                env = os.environ.copy()
                env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
                with (directory/'launcher.log').open('w') as log:
                    process = subprocess.Popen([sys.executable, '-m', __name__, str(directory/'job.json'),
                                                str(lock.fileno()), self.manager.model_dir],
                        cwd=Path(__file__).resolve().parents[2], env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lock.fileno(),))
                self.manager.process = process
                self.manager.directory = directory
            except BaseException:
                for run in job['runs']:
                    if run['id'] in attempts:
                        run.update(status='failed', error='Could not start processing.')
                job['active'] = None
                self.save(job)
                raise
            finally:
                lock.close()  # The child holds the same lock until it exits.
        return self.describe(job_id)


def execute_stage(request):
    """Called inside an isolated, persistent worker on one visible GPU."""
    stage = request.pop('_stage')
    if stage == 'pose':
        from fdanyone.pipeline import extract_pose
        return extract_pose(**request)
    if stage == 'generate':
        from inference import inference
        return inference(**request)
    if stage == 'splat':
        from fdanyone.reconstruction import train_result
        return train_result(**request)
    raise ValueError('Unknown worker stage.')


def run_stage(job, run, model_dir):
    """Execute all chunks of one stage, retaining reports even on failure."""
    directory = Path(job['directory'])
    output = directory/'stages'/run['id']
    output.mkdir(parents=True)
    stage, config = run['stage'], run['config']
    if stage == 'trim':
        from .clips import plan_clip, prepare_chunks
        plan = plan_clip(job['source']['path'], **config)
        videos = prepare_chunks(plan, output/'clips')
        run['artifacts'] = [dict(chunk=i, label=f'Chunk {i+1}', video=video, fps=f"{plan['fps_num']}/{plan['fps_den']}")
                            for i,video in enumerate(videos)]
        return
    predecessor = STAGES[STAGES.index(stage)-1]
    upstream = next(r for r in job['runs'] if r['id'] == run['inputs'][predecessor])
    from fdanyone.streams import run_queue, normalize_gpu_uuid
    from fdanyone.download import ensure_models, ensure_sam3d, ensure_turbo, ensure_foreground_model
    if stage in ('pose', 'generate'):
        models = ensure_models(model_dir) if stage == 'generate' else Path(model_dir).resolve()
        if stage == 'pose':
            ensure_foreground_model(models)
        ensure_sam3d(models)
        if stage == 'generate' and config['turbo']:
            ensure_turbo(models)
    import torch
    gpus = [normalize_gpu_uuid(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
    if not gpus:
        raise ValueError('No CUDA GPU is available.')
    queued = []
    for i, source in enumerate(upstream['artifacts']):
        chunk = output/f'{i:04d}'
        chunk.mkdir()
        artifact = dict(source)
        request = dict(_stage=stage, model_dir=model_dir)
        if stage == 'pose':
            artifact['pose_dir'] = str(chunk/'pose')
            request.update(video_path=source['video'], target_fps=source['fps'], start_time=source.get('start_time',0), output_dir=artifact['pose_dir'], precision=config['precision'])
        elif stage == 'generate':
            from .jobs import make_options
            request.update(make_options(config['views'], config['pitches'], config['yaw'], config['span'], config['turbo'], source.get('start_time',0), source['fps'], config['seed']))
            request.update(video_path=source['video'], pose_dir=source['pose_dir'], data_dir=str(chunk), gpu_ids=[0])
            artifact['result_dir'] = str(chunk/'fdanyone'/Path(source['video']).stem)
        else:
            artifact['training_dir'] = str(chunk/'training')
            request.update(result_dir=source['result_dir'], output_dir=artifact['training_dir'], **config)
        run['artifacts'].append(artifact)
        path = chunk/'request.json'
        write_json(path, request)
        queued.append(dict(request=str(path), log=str(chunk/'worker.log'), video_path=source['video'],
                           result_dir=artifact.get('result_dir', ''), status='queued', label=source['label']))
    write_json(directory/'job.json', job)
    report = run_queue(queued, gpus, output/'streams_report.json')
    if any(row['status'] != 'completed' for row in report['jobs']):
        raise ValueError(f'{LABELS[stage]} failed. Open the task log for details; earlier versions are preserved.')


def run_job(path, model_dir):
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    job = read(path)
    job['directory'] = str(path.parent)
    attempts = job['active']['runs']
    job['active']['pid'] = os.getpid()
    try:
        for run_id in attempts:
            run = next(r for r in job['runs'] if r['id'] == run_id)
            run.update(status='running', started_at=time.time())
            job['active']['stage'] = run['stage']
            write_json(path, job)
            print(f"Starting {LABELS[run['stage']]}", flush=True)
            run_stage(job, run, model_dir)
            run.update(status='completed', completed_at=time.time())
            job['selected'][run['stage']] = run['id']
            job['updated_at'] = time.time()
            write_json(path, job)
    except BaseException as exc:
        for run in job['runs']:
            if run['id'] in attempts and run['status'] in ('queued', 'running'):
                run.update(status='cancelled' if isinstance(exc, (KeyboardInterrupt, SystemExit)) or run['status'] == 'queued' else 'failed', error=str(exc) or 'Stopped by user.')
        import traceback
        traceback.print_exc()
    finally:
        job['active'] = None
        job['updated_at'] = time.time()
        write_json(path, job)


if __name__ == '__main__':
    # The inherited descriptor deliberately remains open for the entire queue.
    run_job(Path(sys.argv[1]), sys.argv[3])
