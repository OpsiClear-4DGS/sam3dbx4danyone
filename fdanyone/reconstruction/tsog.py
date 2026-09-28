"""Package complete Vanilla models with the pinned TSOG v4 reference encoder."""
from fractions import Fraction
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

from fdanyone.errors import ConfigurationError
from .training_preview import write_json

ENCODER = Path(__file__).resolve().parents[2]/'third_party/tsog'
SCRIPT = Path(__file__).with_name('export_tsog.mjs')


def encoder_runtime(*, probe=False):
    node = shutil.which('node')
    if not node or not (ENCODER/'dist/index.mjs').is_file():
        raise ConfigurationError('TSOG export requires Node.js 22+ and the encoder build. Run '
                                 'git submodule update --init --recursive, then '
                                 'npm ci --ignore-scripts --prefix third_party/tsog and '
                                 'npm run build --prefix third_party/tsog.')
    try:
        version = subprocess.check_output([node, '--version'], text=True).strip()
        if int(version.lstrip('v').split('.')[0]) < 22:
            raise ValueError('Node.js 22 or newer is required.')
        if probe:
            subprocess.run([node, str(SCRIPT), '--check'], check=True, capture_output=True, text=True, timeout=60)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, 'stderr', None) or str(exc)
        raise ConfigurationError(f'TSOG encoder check failed (WebGPU/Vulkan is required): {detail[-2000:]}') from exc
    return node


def export_tsog(source, destination, *, fps, audio=None, log=None):
    """Atomically publish a new TSOG; never replace an existing artifact."""
    source, destination = Path(source), Path(destination)
    fps = float(Fraction(str(fps)))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('TSOG FPS must be positive and finite.')
    if destination.suffix != '.tsog':
        raise ValueError('TSOG destination must end in .tsog.')
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    node = encoder_runtime()
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # The reference encoder selects its own WebGPU device, independently of
    # CUDA_VISIBLE_DEVICES. Serialize final packaging across queue workers.
    lock = Path(tempfile.gettempdir())/f'fdanyone-tsog-{os.getuid()}.lock'
    with lock.open('a') as guard, tempfile.TemporaryDirectory(prefix='.tsog-', dir=destination.parent) as temporary:
        fcntl.flock(guard, fcntl.LOCK_EX)
        staged = Path(temporary)/'scene.tsog'
        command = [node, str(SCRIPT), str(source), str(staged), str(fps)]
        if audio is not None:
            command.append(str(audio))
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT if log else None)
        with zipfile.ZipFile(staged) as archive:
            if archive.testzip() is not None:
                raise ValueError('TSOG package has a corrupt entry.')
            meta = json.loads(archive.read('meta.json'))
            if meta.get('version') != 4 or meta.get('timeline', {}).get('type') != 1:
                raise ValueError('TSOG package is not a continuous version-4 model.')
        # Link is atomic and fails if another process published this destination.
        os.link(staged, destination)
    return dict(format='tsog', version=4, count=meta['count'],
                bytes=destination.stat().st_size, source_bytes=source.stat().st_size,
                elapsed_seconds=time.monotonic()-started, motion_bits=16, iterations=10,
                audio=meta.get('audio'), playback=meta.get('playback'),
                encoder_revision='a58102d0489a68c4f7e18e5e1ddaf613256417dd')


def convert_result(result_dir):
    """Upgrade a completed gallery result without retraining or changing its PLY."""
    output = Path(result_dir).expanduser().resolve()/'training'
    status_path = output/'status.json'
    status = json.loads(status_path.read_text())
    if status.get('status') != 'completed':
        raise ConfigurationError('Only completed results can be converted in place.')
    manifest = json.loads((output/'dataset/manifest.json').read_text())
    audio = output.parent/'audio.m4a'
    metrics = export_tsog(output/'scene.ftgs.ply', output/'scene.tsog', fps=manifest['fps'],
                          audio=audio if audio.is_file() else None)
    status.update(model='scene.tsog', container=metrics)
    write_json(status_path, status)
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=convert_result.__doc__)
    parser.add_argument('result_dir', type=Path)
    print(json.dumps(convert_result(parser.parse_args().result_dir), indent=2))
