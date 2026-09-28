"""Decode final views in a fresh CUDA process on the job's assigned device."""
from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fdanyone.config import ExecutionProfile
from fdanyone.device import configure_inference_cuda_allocator
from fdanyone.io import write_json


def publish_isolated(latents, vae_path, output_dir, fps, devices, execution):
    import torch
    devices = tuple(devices[:min(len(devices), len(latents))])
    if len(devices) > 1:
        # One CUDA process per GPU avoids shared cuDNN/allocator state across
        # decoder threads. Keep canonical camera order when collecting files.
        with TemporaryDirectory(prefix='.decode-shards-', dir=output_dir.parent) as temporary:
            root = Path(temporary)
            latents = latents.cpu()
            def decode_shard(slot):
                indices = list(range(slot, len(latents), len(devices)))
                destination = root / str(slot)
                paths, peaks = publish_isolated(
                    latents[indices], vae_path, destination, fps,
                    (devices[slot],), execution)
                return indices, paths, peaks
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                shards = list(pool.map(decode_shard, range(len(devices))))
            video_root = output_dir / 'videos'
            video_root.mkdir(parents=True, exist_ok=False)
            videos = [None] * len(latents)
            peaks = {}
            for indices, paths, shard_peaks in shards:
                if len(indices) != len(paths):
                    raise RuntimeError('Decoder shard returned an incorrect camera count.')
                for index, path in zip(indices, paths, strict=True):
                    destination = video_root / f'{index:02d}.mp4'
                    path.replace(destination)
                    videos[index] = destination
                peaks.update(shard_peaks)
            return tuple(videos), peaks
    if not devices:
        raise ValueError('Isolated decoding requires at least one device and latent view.')
    with TemporaryDirectory(prefix='.decode-', dir=output_dir.parent) as temporary:
        work = Path(temporary)
        torch.save(latents.cpu(), work/'latents.pt')
        request = dict(vae_path=str(vae_path), output_dir=str(output_dir),
                       fps=str(fps), devices=['cuda:0'], execution=asdict(execution))
        write_json(work/'request.json', request)
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
        index = int(devices[0].removeprefix('cuda:'))
        visible = environment.get('CUDA_VISIBLE_DEVICES')
        environment['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index].strip() if visible else str(index)
        subprocess.run([sys.executable, '-m', 'fdanyone.model.decode_worker', str(work)],
                       check=True, env=environment)
        report = json.loads((work/'report.json').read_text())
        return tuple(Path(p) for p in report['videos']), {
            devices[0]: report['peak_vram']['cuda:0']}


def main(work):
    request = json.loads((work/'request.json').read_text())
    execution = ExecutionProfile(**request['execution'])
    configure_inference_cuda_allocator(
        max_split_size_mb=execution.cuda_max_split_size_mb,
        expandable_segments=execution.cuda_expandable_segments,
        garbage_collection_threshold=execution.cuda_garbage_collection_threshold)
    import logging
    import torch
    from fdanyone.model.vae import VaeExecutor
    logging.basicConfig(level=logging.INFO)
    # Initialize device contexts serially before the threaded VAE workers.
    # Lazy concurrent initialization with expandable segments can fail in
    # model.to(device) with CUDA "operation not permitted".
    # Queue workers already carry a CPU budget. Keep the historical eight-thread
    # fallback only for direct decoder calls with no explicit thread settings.
    if not (os.environ.get('OMP_NUM_THREADS') or os.environ.get('MKL_NUM_THREADS')):
        torch.set_num_threads(8)
    for device in request['devices']:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
    torch.cuda.set_device(request['devices'][0])
    latents = torch.load(work/'latents.pt', map_location='cpu', weights_only=True)
    executor = VaeExecutor.load(Path(request['vae_path']), tuple(request['devices']), execution)
    try:
        videos = executor.publish_targets(latents, Path(request['output_dir']),
                                          SimpleNamespace(fps=Fraction(request['fps'])))
        write_json(work/'report.json', dict(videos=[str(p) for p in videos],
                                           peak_vram=executor.last_peak_vram_bytes))
    finally:
        executor.close()


if __name__ == '__main__':
    main(Path(sys.argv[1]))
