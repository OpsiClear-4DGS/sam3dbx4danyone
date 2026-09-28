"""Export generated views and SAM mesh motion to the FreeTimeGS data contract."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import av
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation

from fdanyone.errors import ConfigurationError
from fdanyone.io import AtomicResultDirectory, write_json


def read_result(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    metadata = json.loads((root/'metadata.json').read_text())['output']
    rig = json.loads((root/'cameras.json').read_text())
    cameras = rig['cameras']
    if rig.get('camera_model') != 'OPENCV' or len(cameras) < 2:
        raise ConfigurationError('4DGS training requires at least two calibrated OpenCV views.')
    if [c['camera_id'] for c in cameras] != list(range(len(cameras))):
        raise ConfigurationError('Camera IDs must match the ordered generated views.')
    frames = int(metadata['frames_per_video'])
    if frames < 2:
        raise ConfigurationError('4DGS training requires at least two frames.')
    for camera in cameras:
        video = (root/camera['video']).resolve(strict=True)
        if not video.is_relative_to(root) or not video.is_file():
            raise ConfigurationError('Generated videos must be files inside the result directory.')
        K = np.asarray(camera['K'], dtype=np.float64)
        c2w = np.asarray(camera['camera_to_world'], dtype=np.float64)
        if (K.shape != (3, 3) or c2w.shape != (4, 4)
                or not np.isfinite(K).all() or not np.isfinite(c2w).all()
                or not np.allclose(c2w[3], [0, 0, 0, 1])
                or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(c2w[:3, :3]), 1)
                or not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0
                or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise ConfigurationError('Invalid pinhole intrinsics or rigid camera transform.')
        if (camera['image_width'], camera['image_height']) != (metadata['width'], metadata['height']):
            raise ConfigurationError('Camera and generated video resolutions do not match.')
    with np.load(root/'preprocessing/viewer_geometry.npz', allow_pickle=False) as geometry:
        if 'vertices' not in geometry or 'faces' not in geometry:
            raise ConfigurationError('This result has no animated SAM mesh. Regenerate it with body geometry enabled.')
        vertices = geometry['vertices'].astype(np.float32)
        faces = geometry['faces']
    if (vertices.ndim != 3 or vertices.shape[0] != frames or vertices.shape[2] != 3
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer) or not faces.size
            or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ConfigurationError('Invalid SAM mesh topology or animation length.')
    return root, metadata, cameras, vertices, faces


def mesh_initialization(vertices, faces, *, samples=32768, keyframe_step=5):
    """Area-sampled surface seeds; topology gives motion in meters per frame.

    Resample each keyframe to avoid coincident seeds in stationary body regions.
    Each sample's triangle and barycentric coordinates remain fixed when taking
    its central velocity difference. Include the final frame exactly once.
    """
    if samples < 4 or keyframe_step < 1:
        raise ConfigurationError('Use at least four surface samples and a positive keyframe step.')
    count = len(vertices)
    keys = sorted(set(range(0, count, keyframe_step)) | {count-1})
    rng = np.random.default_rng(42)
    positions, velocities, normals = [], [], []
    for frame in keys:
        triangles = vertices[frame][faces]
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1)
        if areas.sum() <= 0:
            raise ConfigurationError('The SAM mesh contains no nondegenerate surface.')
        selected = rng.choice(len(faces), samples, p=areas.astype(np.float64)/areas.sum(dtype=np.float64))
        bary = rng.random((samples, 2)).astype(np.float32)
        bary[bary.sum(axis=1) > 1] = 1-bary[bary.sum(axis=1) > 1]
        bary = np.column_stack((1-bary.sum(axis=1), bary))
        def sample(index):
            return np.einsum('nij,ni->nj', vertices[index][faces[selected]], bary)
        before, after = max(0, frame-1), min(count-1, frame+1)
        positions.append(sample(frame))
        velocities.append((sample(after)-sample(before))/(after-before))
        normals.append(cross[selected]/np.maximum(areas[selected, None], 1e-12))
    return keys, np.asarray(positions), np.asarray(velocities), np.asarray(normals)


def project_colors(points, normals, camera, image, mask):
    """Select visible surface colors using coarse depth and viewing angle.

    This is an initialization heuristic; the trainer refines appearance and
    geometry against all masked images. No image is resized or re-cropped.
    """
    w2c = np.linalg.inv(np.asarray(camera['camera_to_world']))
    xyz = points @ w2c[:3, :3].T + w2c[:3, 3]
    projected = xyz @ np.asarray(camera['K']).T
    xy = np.rint(projected[:, :2]/np.maximum(xyz[:, 2:3], 1e-8)).astype(np.int64)
    h, w = image.shape[:2]
    valid = (xyz[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    indices = np.flatnonzero(valid)
    x, y = xy[indices].T
    depth = np.full(((h+3)//4, (w+3)//4), np.inf)
    np.minimum.at(depth, (y//4, x//4), xyz[indices, 2])
    nearest = minimum_filter(depth, size=3)[y//4, x//4]
    direction = np.asarray(camera['camera_to_world'])[:3, 3]-points[indices]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
    weights = np.abs(np.einsum('ij,ij->i', normals[indices], direction))
    weights *= (mask[y, x] >= 128) & (xyz[indices, 2] <= nearest + .025)
    return indices, image[y, x].astype(np.float32)/255, weights


def write_colmap(directory, cameras, points, colors):
    """One reference image per fixed camera; all video frames live in its folder."""
    directory.mkdir(parents=True)
    # Binary also supports the pinned legacy pycolmap reader, whose text reader
    # stops on empty POINTS2D lines instead of retaining pose-only images.
    with (directory/'cameras.bin').open('wb') as calibration, (directory/'images.bin').open('wb') as images:
        calibration.write(struct.pack('<Q', len(cameras)))
        images.write(struct.pack('<Q', len(cameras)))
        for index, camera in enumerate(cameras):
            K = np.asarray(camera['K'])
            w2c = np.linalg.inv(np.asarray(camera['camera_to_world']))
            xyzw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
            pose = [xyzw[3], *xyzw[:3], *w2c[:3, 3]]
            calibration.write(struct.pack('<iiQQ4d', index+1, 1, camera['image_width'], camera['image_height'],
                                          K[0,0], K[1,1], K[0,2], K[1,2]))
            images.write(struct.pack('<i7di', index+1, *pose, index+1))
            images.write(f'cam{index:03d}/000000.png'.encode()+b'\0'+struct.pack('<Q', 0))
    with (directory/'points3D.bin').open('wb') as output:
        output.write(struct.pack('<Q', len(points)))
        for index, (point, color) in enumerate(zip(points, colors)):
            output.write(struct.pack('<Q3d3BdQ', index+1, *point,
                                     *np.rint(color*255).clip(0,255).astype(np.uint8), 0., 0))


def prepare_dataset(result_dir, destination, segmenter, *, samples=32768, progress=None):
    root, metadata, cameras, vertices, faces = read_result(result_dir)
    keys, positions, velocities, normals = mesh_initialization(vertices, faces, samples=samples)
    key_index = {frame: index for index, frame in enumerate(keys)}
    colors = np.full_like(positions, .5)
    best = np.zeros(positions.shape[:2], dtype=np.float32)
    frames = metadata['frames_per_video']
    with AtomicResultDirectory(destination) as work:
        for view, camera in enumerate(cameras):
            folder = work/'images'/f'cam{view:03d}'
            folder.mkdir(parents=True)
            processed = 0
            def consume(batch):
                nonlocal processed
                masks = segmenter(tuple(batch))
                if masks.shape != (len(batch), *batch[0].shape[:2]) or masks.dtype != np.uint8:
                    raise ConfigurationError('BiRefNet returned invalid training masks.')
                for image, mask in zip(batch, masks):
                    if not (mask >= 128).any():
                        raise ConfigurationError(f'Empty foreground in camera {view}, frame {processed}.')
                    # Straight RGBA: retain both the original RGB and soft alpha.
                    # Black RGB alone cannot distinguish clothing from empty space.
                    rgba = np.concatenate((image, mask[..., None]), axis=-1)
                    Image.fromarray(rgba).save(folder/f'{processed:06d}.png', compress_level=1)
                    if processed in key_index:
                        k = key_index[processed]
                        idx, rgb, weights = project_colors(positions[k], normals[k], camera, image, mask)
                        keep = weights > best[k, idx]
                        colors[k, idx[keep]] = rgb[keep]
                        best[k, idx[keep]] = weights[keep]
                    processed += 1
            with av.open(str(root/camera['video'])) as video:
                batch = []
                for frame in video.decode(video=0):
                    image = frame.to_ndarray(format='rgb24')
                    if image.shape[:2] != (metadata['height'], metadata['width']):
                        raise ConfigurationError('Decoded raster differs from camera calibration.')
                    if processed + len(batch) >= frames:
                        raise ConfigurationError('Generated video has more frames than its metadata.')
                    batch.append(image)
                    if len(batch) == 4:
                        consume(batch)
                        batch = []
                if batch:
                    consume(batch)
            if processed != frames:
                raise ConfigurationError(f'Camera {view} has {processed} frames; expected {frames}.')
            if progress:
                progress(view+1, len(cameras))
        span = frames-1
        np.savez(work/'init.npz', positions=positions.reshape(-1, 3), velocities=velocities.reshape(-1, 3),
                 colors=colors.reshape(-1, 3), times=np.repeat(np.asarray(keys, np.float32)/span, samples)[:, None],
                 durations=np.full((len(keys)*samples, 1), 2*min(5, span)/span, np.float32),
                 has_velocity=np.ones(len(keys)*samples, dtype=bool), frame_start=0, frame_end=frames,
                 time_denominator=span, keyframe_step=5, n_keyframes=len(keys), mode='sam_mesh_velocity')
        write_colmap(work/'sparse/0', cameras, positions[0], colors[0])
        manifest = dict(source=str(root), frames=frames, cameras=len(cameras), fps=metadata['fps'],
                        width=metadata['width'], height=metadata['height'], keyframes=keys,
                        initial_gaussians=len(keys)*samples, colored_fraction=float(np.mean(best > 0)),
                        initialization='SAM mesh surface with topology-derived velocities (meters/frame)',
                        format_version=2, alpha_mode='transparent',
                        foreground='BiRefNet soft alpha, supervised transparency',
                        image_format='lossless straight RGBA PNG')
        write_json(work/'manifest.json', manifest)
    return manifest
