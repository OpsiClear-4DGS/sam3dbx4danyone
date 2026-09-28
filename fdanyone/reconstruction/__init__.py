"""Isolated FreeTimeGsVanilla training after multi-view generation."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from fdanyone.errors import ConfigurationError
from fdanyone.io import write_json

REPO = Path(__file__).resolve().parents[2]/'third_party/FreeTimeGsVanilla'
WORKER = Path(__file__).with_name('train_worker.py')
DEFAULT_STEPS = 30_000


def runtime():
    if not (REPO/'src/simple_trainer_freetime_4d_pure_relocation.py').is_file():
        raise ConfigurationError('Initialize FreeTimeGsVanilla: git submodule update --init --recursive')
    # Do not resolve the executable symlink: Python locates its venv beside it.
    python = Path(os.environ.get('FDANYONE_FREETIMEGS_PYTHON', str(REPO/'.venv/bin/python'))).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ConfigurationError('Set up training with uv sync --locked --project third_party/FreeTimeGsVanilla, '
                                 'or set FDANYONE_FREETIMEGS_PYTHON to its existing environment Python.')
    return python


def worker_environment(gpu_uuid=None):
    environment = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PYTORCH_CUDA_ALLOC_CONF',
                'PYTORCH_ALLOC_CONF', 'FDANYONE_REUSE_DENOISER'):
        environment.pop(key, None)
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('OMP_NUM_THREADS', '8')
    environment.setdefault('MKL_NUM_THREADS', '8')
    if gpu_uuid:
        environment['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    return environment


def check_runtime(*, probe=False):
    python = runtime()
    if probe:
        with tempfile.TemporaryDirectory(prefix='fdanyone-ftgs-check-') as temporary:
            request = Path(temporary)/'check.json'
            write_json(request, {'repo': str(REPO), 'check': True})
            try:
                result = subprocess.run([str(python), str(WORKER), str(request)], cwd=REPO,
                                        env=worker_environment(), capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ConfigurationError(f'Cannot start the FreeTimeGS environment check: {exc}') from exc
            if result.returncode:
                raise ConfigurationError('FreeTimeGS environment check failed:\n'+result.stderr[-3000:])
    return python


def training_status(result_dir):
    try:
        value = json.loads((Path(result_dir)/'training/status.json').read_text())
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


@contextmanager
def _interruptible():
    previous = signal.getsignal(signal.SIGTERM)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def train_result(result_dir, *, model_dir='models', output_dir=None, gpu_id=0,
                 steps=DEFAULT_STEPS, samples_per_keyframe=32768):
    """Train a foreground 4D scene; keep generation output intact on failure."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ConfigurationError('training_steps must be a positive integer.')
    if isinstance(samples_per_keyframe, bool) or not isinstance(samples_per_keyframe, int) or samples_per_keyframe < 4:
        raise ConfigurationError('samples_per_keyframe must be an integer of at least four.')
    python = check_runtime(probe=True)
    from .dataset import prepare_dataset, read_result
    result, _, _, _, _ = read_result(result_dir)
    output = Path(output_dir).expanduser().absolute() if output_dir else result/'training'
    if os.path.lexists(output):
        raise ConfigurationError(f'Training output already exists: {output}. Choose a new output_dir to retain prior runs.')
    from fdanyone.device import configure_inference_cpu_environment, default_cpu_threads
    configure_inference_cpu_environment()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', default_cpu_threads())))
    from fdanyone.model.loader import clear_denoiser_cache
    from fdanyone.streams import normalize_gpu_uuid
    clear_denoiser_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gpu_uuid = normalize_gpu_uuid(torch.cuda.get_device_properties(gpu_id).uuid)
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    # Record exact sources, including any local changes to the pinned trainer.
    source_hashes = {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in (
        'src/simple_trainer_freetime_4d_pure_relocation.py', 'src/foreground_loss.py',
        'datasets/FreeTime_dataset.py', 'src/freetime_ops.py')}
    output.mkdir(parents=True)
    status_path = output/'status.json'
    started = time.monotonic()
    status = dict(status='running', stage='preparing training views', steps=steps, step=0,
                  trainer_revision=revision, trainer_source_sha256=source_hashes,
                  alpha_mode='transparent', gpu_uuid=gpu_uuid, result_dir=str(result),
                  output_dir=str(output), started_at=time.time())
    def update(**values):
        # Worker progress fields may be newer than this process's copy.
        if status_path.exists():
            status.update(json.loads(status_path.read_text()))
        status.update(values, elapsed_seconds=time.monotonic()-started)
        write_json(status_path, status)
    update()
    try:
        with _interruptible():
            from fdanyone.download import ensure_foreground_model
            from fdanyone.foreground import ForegroundSegmenter
            with ForegroundSegmenter(ensure_foreground_model(model_dir), f'cuda:{gpu_id}') as segmenter:
                manifest = prepare_dataset(result, output/'dataset', segmenter,
                    samples=samples_per_keyframe,
                    progress=lambda done, total: update(prepared_views=done, total_views=total))
            request = output/'request.json'
            write_json(request, dict(repo=str(REPO), dataset=str(output/'dataset'), output=str(output),
                                     steps=steps, status=str(status_path)))
            update(stage='training', dataset=manifest)
            print(f'4DGS training: {steps:,} steps; log: {output / "train.log"}', flush=True)
            with (output/'train.log').open('w') as log:
                subprocess.run([str(python), str(WORKER), str(request)], cwd=REPO,
                               env=worker_environment(gpu_uuid), stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (output/'scene.ftgs.ply').is_file() or not (output/'ckpts'/f'ckpt_{steps-1}.pt').is_file():
                raise ConfigurationError('FreeTimeGS finished without a checkpoint and animated model.')
            update(status='completed', stage='completed', step=steps, model='scene.ftgs.ply',
                   checkpoint=f'ckpts/ckpt_{steps-1}.pt')
            from .training_preview import clear_previews
            clear_previews(output)
    except BaseException as exc:
        cancelled = isinstance(exc, (KeyboardInterrupt, SystemExit))
        update(status='cancelled' if cancelled else 'failed', error=str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            raise ConfigurationError(f'4DGS training failed; generated videos are preserved. See {output / "train.log"}') from exc
        raise
    return status
