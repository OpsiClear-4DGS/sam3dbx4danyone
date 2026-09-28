"""Read calibrated SAM scenes and prepare browser-compatible video copies."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import threading
import uuid

import av
import numpy as np
from PIL import Image


_EXPORT_LOCK = threading.Lock()


def contained_file(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or external result file: {relative}')
    return path


def read_result(directory):
    root = Path(directory).expanduser().resolve()
    metadata = json.loads(contained_file(root, 'metadata.json').read_text())
    rig = json.loads(contained_file(root, 'cameras.json').read_text())
    cameras = rig['cameras']
    if rig.get('camera_model') != 'OPENCV' or not cameras:
        raise ValueError('Expected an OpenCV camera rig.')
    if [c['camera_id'] for c in cameras] != list(range(len(cameras))):
        raise ValueError('Camera IDs must be in canonical order.')
    if rig['world_frame']['name'] != 'canonical_human_world' or rig['camera_frame']['name'] != 'opencv_camera':
        raise ValueError('Unsupported camera coordinates.')
    videos = [contained_file(root, c['video']) for c in cameras]
    fps = Fraction(metadata['output']['fps'])
    if fps <= 0 or metadata['output']['frames_per_video'] != 121 or metadata['output']['target_views'] != len(cameras):
        raise ValueError('Expected synchronized 121-frame outputs.')
    for c in cameras:
        k, transform = np.asarray(c['K']), np.asarray(c['camera_to_world'])
        if (k.shape != (3, 3) or transform.shape != (4, 4) or not np.isfinite(k).all()
                or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])
                or min(c['image_width'], c['image_height']) <= 0):
            raise ValueError('Invalid camera calibration.')
    return root, metadata, cameras, videos, fps


def result_thumbnail(directory, cache_dir):
    """Cache a small three-angle contact sheet without exporting playback videos."""
    _, _, _, videos, _ = read_result(directory)
    sources = [videos[i] for i in sorted({0, len(videos)//3, 2*len(videos)//3})]
    identities = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    key = hashlib.sha256(json.dumps(identities).encode()).hexdigest()[:24]
    cache = Path(cache_dir)/'gallery'
    if cache.resolve() != cache:
        raise ValueError('Invalid gallery cache directory.')
    target = cache/(key+'.jpg')
    if target.is_symlink():
        raise ValueError('Invalid gallery thumbnail.')
    if target.is_file():
        return target
    canvas = Image.new('RGB', (480, 288), '#202020')
    cell_width = canvas.width//len(sources)
    for i, source in enumerate(sources):
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            # Decode only the opening frame; gallery browsing never transcodes
            # whole clips or loads models, body geometry, or GPU resources.
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise ValueError('This result has no video frames.')
            image = frame.to_image()
            image.thumbnail((cell_width, canvas.height), Image.Resampling.LANCZOS)
            canvas.paste(image, (i*cell_width+(cell_width-image.width)//2,
                                 (canvas.height-image.height)//2))
    cache.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.'+uuid.uuid4().hex+'.jpg')
    try:
        canvas.save(temporary, format='JPEG', quality=82)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def playback_copy(source, destination, fps, dimensions):
    """Convert RGB-lossless originals into compatible H.264 browser previews."""
    if destination.exists():
        return destination
    temporary = destination.with_name('.' + uuid.uuid4().hex + '.mp4')
    try:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '2',
                        '-i', str(source), '-an', '-c:v', 'libx264', '-threads', '2',
                        '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(temporary)], check=True)
        with av.open(str(temporary)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if stream.average_rate != fps or (stream.width, stream.height) != dimensions:
                raise ValueError('Playback dimensions or FPS disagree with metadata.')
            if sum(1 for _ in container.decode(video=0)) != 121:
                raise ValueError('Playback must contain 121 frames.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def scene_payload(cameras, fps=25, points=None):
    from fdanyone.skeleton.keypoints import LINKS, VISIBLE_KEYPOINT_IDS, keypoint_color
    payload = dict(cameras=cameras, fps=float(fps), frames=121 if points is not None else 1,
                   keypoints=None, links=[], joint_ids=[], joint_colors=[])
    if points is not None:
        if points.shape != (121, 70, 3) or not np.isfinite(points).all():
            raise ValueError('Invalid canonical SAM skeleton.')
        payload.update(keypoints=points.tolist(),
                       links=[dict(a=a, b=b, color=list(color)) for _, a, b, color, _ in LINKS],
                       joint_ids=sorted(VISIBLE_KEYPOINT_IDS),
                       joint_colors=[list(keypoint_color(j)) for j in sorted(VISIBLE_KEYPOINT_IDS)])
    return payload


def load_points(geometry):
    if not geometry.exists():
        return None
    with np.load(geometry, allow_pickle=False) as data:
        return data['keypoints']


def export_mesh(geometry, cache):
    """Cache float32 vertices and uint32 topology without expanding them into JSON.

    The little-endian MHR1 header is followed by [frames, vertices, 3] positions
    and [faces, 3] indices. Older joint-only recordings remain supported.
    """
    if not geometry.exists():
        return None
    identity = f'{geometry}:{geometry.stat().st_size}:{geometry.stat().st_mtime_ns}'
    target = Path(cache)/('body-' + hashlib.sha256(identity.encode()).hexdigest()[:16] + '.bin')
    if not target.exists():
        with np.load(geometry, allow_pickle=False) as data:
            if 'vertices' not in data and 'faces' not in data:
                return None
            vertices, faces = data['vertices'], data['faces']
            if (vertices.ndim != 3 or vertices.shape[0] != 121 or vertices.shape[2] != 3
                    or not 3 <= vertices.shape[1] <= 100000 or not np.isfinite(vertices).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or not 1 <= len(faces) <= 200000
                    or not np.issubdtype(faces.dtype, np.integer)
                    or faces.min() < 0 or faces.max() >= vertices.shape[1]):
                raise ValueError('Invalid SAM body mesh.')
            temporary = target.with_name('.' + uuid.uuid4().hex + '.bin')
            try:
                with temporary.open('wb') as stream:
                    stream.write(struct.pack('<4sIII', b'MHR1', len(vertices), vertices.shape[1], len(faces)))
                    np.asarray(vertices, dtype='<f4').tofile(stream)
                    np.asarray(faces, dtype='<u4').tofile(stream)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return {'url': str(target)}


def export_result(directory, cache_dir):
    root, metadata, cameras, videos, fps = read_result(directory)
    identities = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in [root/'metadata.json', root/'cameras.json', *videos]]
    geometry = root/'preprocessing/viewer_geometry.npz'
    if geometry.exists():
        geometry = contained_file(root, 'preprocessing/viewer_geometry.npz')
        identities.append((str(geometry), geometry.stat().st_size, geometry.stat().st_mtime_ns))
    key = hashlib.sha256(json.dumps(identities).encode()).hexdigest()[:24]
    cache = Path(cache_dir)/key
    with _EXPORT_LOCK:
        # Adopt the matching pre-history cache without transcoding again. The
        # key includes the absolute result path, so it belongs to this job only.
        legacy = Path(cache_dir).parent/key
        if not cache.exists() and legacy.is_dir() and not legacy.is_symlink():
            cache.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(cache)
        cache.mkdir(parents=True, exist_ok=True)
        outputs = [cache/f'{i:02d}.mp4' for i in range(len(videos))]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(playback_copy, p, outputs[i], fps,
                       (cameras[i]['image_width'], cameras[i]['image_height'])) for i, p in enumerate(videos)]
            for future in futures:
                future.result()
        payload = scene_payload(cameras, fps, load_points(geometry))
        payload['mesh'] = export_mesh(geometry, cache)
        payload['frames'] = 121
    note = ('SAM body mesh and calibrated cameras.' if payload['mesh'] else
            'Canonical SAM skeleton and calibrated cameras.' if geometry.exists() else
            'Older result: calibrated cameras; no saved SAM skeleton.')
    return payload, [str(p) for p in outputs], metadata, note


def export_layout(options):
    from fdanyone.geometry.cameras import camera_grid, reference_intrinsics
    cameras = camera_grid(center=np.zeros(3), front_direction=np.array([0., 0., 1.]),
                          K=reference_intrinsics(1280, 704), image_height=1280, image_width=704,
                          **{k: options[k] for k in ('views_per_layer', 'layer_pitches', 'start_yaw', 'yaw_span')})
    return scene_payload([c.to_dict() for c in cameras])


def live_preview(job, cache_dir, previous_version=''):
    """Read only completed preprocessing and muxed videos; never change GPU work."""
    data = Path(job['request']).parent
    for scratch in sorted(data.glob('.sam3d-*')):
        prep = scratch/'preprocessing'
        try:
            cameras = json.loads((prep/'conditioning/cameras.json').read_text())['cameras']
            meta = json.loads((prep/'conditioning/metadata.json').read_text())
            geometry = prep/'viewer_geometry.npz'
            if not geometry.is_file():
                continue
            fps = Fraction(meta['fps_num'], meta['fps_den'])
            sources = []
            for camera in cameras:
                source = scratch/'generation/target/videos'/f"{camera['camera_id']:02d}.mp4"
                try:
                    with av.open(str(source)) as container:
                        stream = container.streams.video[0]
                        ready = stream.frames == 121 and stream.average_rate == fps
                    sources.append(source if ready else None)
                except (av.error.FFmpegError, OSError, IndexError):
                    sources.append(None)
            identity = [str(scratch), geometry.stat().st_mtime_ns,
                        [(str(p), p.stat().st_size) if p else None for p in sources]]
            version = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
            if version == previous_version:
                return {'version': version, 'unchanged': True}
            payload = scene_payload(cameras, fps, load_points(geometry))
            cache = Path(cache_dir)/('live-' + hashlib.sha256(str(scratch).encode()).hexdigest()[:24])
            cache.mkdir(parents=True, exist_ok=True)
            videos = []
            with _EXPORT_LOCK:
                payload['mesh'] = export_mesh(geometry, cache)
                for i, source in enumerate(sources):
                    videos.append(str(playback_copy(source, cache/f'{i:02d}.mp4', fps,
                                  (cameras[i]['image_width'], cameras[i]['image_height']))) if source else None)
            return dict(version=version, scene=payload, videos=videos,
                        note=f'SAM body ready · {sum(p is not None for p in sources)}/{len(cameras)} generated cameras ready')
        except (OSError, ValueError, KeyError):
            # The worker may atomically finish and remove scratch while we read.
            continue
    return {'pending': True}
