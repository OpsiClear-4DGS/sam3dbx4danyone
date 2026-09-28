"""Run one independent video job per visible GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from fdanyone.errors import ConfigurationError
from fdanyone.io import write_json


def normalize_gpu_uuid(uuid):
    value = str(uuid)
    return value if value.startswith(('GPU-', 'MIG-')) else f'GPU-{value}'


def worker_environment(gpu_uuid, cpu_threads=None):
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
    environment.pop('PYTHONHOME', None)
    from fdanyone.device import default_cpu_threads
    threads = (environment.get('OMP_NUM_THREADS') or environment.get('MKL_NUM_THREADS')
               or str(cpu_threads or default_cpu_threads()))
    environment.setdefault('OMP_NUM_THREADS', threads)
    environment.setdefault('MKL_NUM_THREADS', threads)
    environment.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
    return environment


def run_queue(jobs, gpus, report_path):
    """Keep one worker per GPU alive so queued videos reuse compiled models."""
    active = {}
    workers = {}
    pending = iter(jobs)
    report = {'gpus': gpus, 'jobs': jobs}
    started = time.monotonic()
    completed_normally = False
    from fdanyone.device import default_cpu_threads
    cpu_threads = default_cpu_threads(min(len(gpus), len(jobs)))
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(report_path, report)
    def launch(slot, gpu_uuid):
        with open(Path(report_path).parent / f'gpu-{slot}.log', 'ab') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'fdanyone.streams', '--worker'],
                env=worker_environment(gpu_uuid, cpu_threads), stdin=subprocess.PIPE,
                stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=True)
        workers[slot] = process
        return process
    save()
    try:
        while True:
            for slot, gpu_uuid in enumerate(gpus):
                process = workers.get(slot)
                if slot in active:
                    job, launched, completion = active[slot]
                    if completion.is_file():
                        code = int(json.loads(completion.read_text())['returncode'])
                    elif process.poll() is not None:
                        code = process.returncode or 1
                    else:
                        continue
                    job.update(status='completed' if code == 0 else 'failed',
                               returncode=code, elapsed_seconds=time.monotonic()-launched)
                    del active[slot]
                    if code != 0 and process.poll() is None:
                        # A CUDA error may poison the context. Restart the slot
                        # after any failure instead of contaminating later jobs.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    save()
                job = next(pending, None)
                if job is None:
                    continue
                if process is None or process.poll() is not None:
                    if process is not None:
                        process.stdin.close()
                    process = launch(slot, gpu_uuid)
                completion = Path(job['request']).with_suffix('.completion.json')
                completion.unlink(missing_ok=True)
                job.update(status='running', slot=slot, gpu_uuid=gpu_uuid, pid=process.pid)
                active[slot] = (job, time.monotonic(), completion)
                try:
                    process.stdin.write(json.dumps(dict(request=job['request'], log=job['log'],
                                                        completion=str(completion))) + '\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    # The normal exit check marks this job failed on the next poll.
                    process.wait()
                save()
            if not active:
                break
            time.sleep(0.2)
        completed_normally = True
    finally:
        for process in workers.values():
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        for process in workers.values():
            if completed_normally:
                try:
                    process.wait(timeout=5)
                    continue
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if not completed_normally and workers:
            time.sleep(0.5)
        for process in workers.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        for job in jobs:
            if job['status'] in ('queued', 'running'):
                job['status'] = 'cancelled'
        save()
    return report


def _serve_worker():
    """Read one job at a time; isolate logs and discard cached state on failure."""
    import traceback
    os.environ['FDANYONE_REUSE_DENOISER'] = '1'
    from inference import inference
    for line in sys.stdin:
        job = json.loads(line)
        request = Path(job['request'])
        code = 0
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        try:
            with open(job['log'], 'w') as log:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                try:
                    result = inference(**json.loads(request.read_text()))
                    write_json(request.with_name('result.json'), result)
                except Exception:
                    code = 1
                    traceback.print_exc()
                    from fdanyone.model.loader import clear_denoiser_cache
                    clear_denoiser_cache()
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
        finally:
            for target, original in zip((1, 2), saved):
                os.dup2(original, target)
                os.close(original)
        write_json(job['completion'], {'returncode': code})


def batch_inference(video_paths: list[str],
                    output_dir: str = 'data/streams', model_dir: str = 'models',
                    **inference_options) -> dict:
    """Generate videos concurrently, one job on each visible GPU.

    video_paths: List of local video files (each needs 121 usable frames).
    output_dir: New directory for the queue report, logs and per-job results.
    inference_options: Normal inference.py options, e.g. turbo=True,
        views_per_layer=6, execution_profile='full'.
    """
    import inspect
    from inference import inference
    if not isinstance(video_paths, (list, tuple)) or not video_paths:
        raise ConfigurationError('video_paths must be a non-empty list of local video paths.')
    reserved = {'video_path', 'data_dir', 'model_dir', 'gpu_ids'}
    invalid = set(inference_options) - (set(inspect.signature(inference).parameters) - reserved)
    if invalid:
        raise ConfigurationError(f'Unsupported or reserved inference options: {sorted(invalid)}')
    paths = [Path(p).expanduser().resolve() for p in video_paths]
    for path in paths:
        if not path.is_file():
            raise ConfigurationError(f'Input video does not exist: {path}')
    import torch
    visible = [normalize_gpu_uuid(torch.cuda.get_device_properties(i).uuid)
               for i in range(torch.cuda.device_count())]
    if not visible:
        raise ConfigurationError('Multi-stream inference requires at least one visible CUDA GPU.')
    output = Path(output_dir).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ConfigurationError(f'Output directory already exists: {output}. Choose a new output_dir.') from None
    # Prepare shared assets serially before children can request them concurrently.
    from fdanyone.download import ensure_models, ensure_sam3d, ensure_turbo
    models = ensure_models(model_dir)
    ensure_sam3d(models)
    if inference_options.get('turbo', True):
        ensure_turbo(models)
    jobs = []
    for index, path in enumerate(paths):
        root = output / f'{index:04d}_{path.stem}'
        root.mkdir()
        request = root / 'request.json'
        write_json(request, dict(inference_options, video_path=str(path),
                                model_dir=str(models), data_dir=str(root), gpu_ids=[0]))
        jobs.append(dict(video_path=str(path), request=str(request), log=str(root/'worker.log'),
                         result_dir=str(root/'fdanyone'/path.stem), status='queued'))
    report = run_queue(jobs, visible, output/'streams_report.json')
    if any(job['status'] != 'completed' for job in jobs):
        raise ConfigurationError(f'Some streams failed; see {output}/streams_report.json and worker logs.')
    return report


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        _serve_worker()
    else:
        from inference import inference
        result = inference(**json.loads(Path(sys.argv[1]).read_text()))
        write_json(Path(sys.argv[1]).with_name('result.json'), result)
