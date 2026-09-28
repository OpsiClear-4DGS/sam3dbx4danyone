# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Publish generated videos and their camera metadata."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fdanyone.config import INFERENCE
from fdanyone.device import CUDA_ALLOCATOR_CONF
from fdanyone.errors import FourDAnyoneError
from fdanyone.io import link_or_copy_file, write_json

if TYPE_CHECKING:
    from fdanyone.model.inference import GeneratedViews
    from fdanyone.skeleton.pipeline import Conditioning
    from fdanyone.video import CanonicalClip


def _copy_file(source: Path, destination: Path) -> Path:
    return link_or_copy_file(source, destination)


def _runtime_metadata(device: str) -> dict:
    import torch

    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "allocator_config": os.environ.get(CUDA_ALLOCATOR_CONF),
    }
    if torch.cuda.is_available():
        torch_device = torch.device(device)
        properties = torch.cuda.get_device_properties(torch_device)
        cuda.update(
            {
                "device": device,
                "device_name": torch.cuda.get_device_name(torch_device),
                "device_capability": list(torch.cuda.get_device_capability(torch_device)),
                "device_total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda": cuda,
    }


def _camera_rig_payload(payload: dict, cameras: list[dict]) -> dict:
    """Keep the final OpenCV camera rig needed by downstream tools."""

    records = []
    for camera in cameras:
        camera_id = int(camera["camera_id"])
        records.append(
            {
                "camera_id": camera_id,
                "layer_index": int(camera["layer_index"]),
                "pitch": int(camera["pitch_degrees"]),
                "yaw": float(camera["yaw_degrees"]),
                "K": camera["K"],
                "camera_to_world": camera["camera_to_world"],
                "image_width": int(camera["image_width"]),
                "image_height": int(camera["image_height"]),
                "video": f"videos/dense/{camera_id:02d}.mp4",
                "skeleton_video": f"skeletons/{camera_id:02d}.mp4",
            }
        )
    return {
        "camera_model": "OPENCV",
        "world_frame": payload["world_frame"],
        "camera_frame": payload["camera_frame"],
        "front_camera_ids": payload["front_camera_ids"],
        "framing": payload["framing"],
        "cameras": records,
    }


def _target_cameras(payload: object, expected_count: int) -> list[dict]:
    """Read the camera records produced by the conditioning stage."""

    if not isinstance(payload, dict) or payload.get("camera_model") != "OPENCV":
        raise FourDAnyoneError("Conditioning did not produce an OpenCV camera rig.")
    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != expected_count:
        raise FourDAnyoneError(f"Conditioning must contain {expected_count} target cameras.")
    if [camera.get("camera_id") for camera in cameras if isinstance(camera, dict)] != list(range(expected_count)):
        raise FourDAnyoneError("Target cameras are not in canonical order.")
    return cameras


def export_result(
    *,
    clip: CanonicalClip,
    conditioning: Conditioning,
    generated: GeneratedViews,
    destination: str | Path,
    motion: dict,
    model_identity: dict,
    pipeline_started: float,
) -> dict:
    """Publish proposal, target, skeleton, camera, and metadata artifacts."""

    from fdanyone.vendor.diffsynth.models.wan_video_dit import get_attention_backend

    root = Path(destination).expanduser().resolve()
    attention_backend = get_attention_backend()
    view_plan = generated.view_plan
    if conditioning.view_plan != view_plan:
        raise FourDAnyoneError("Conditioning and generation resolved different view plans.")
    if len(generated.rcp_videos) != len(view_plan.rcp_camera_ids):
        raise FourDAnyoneError(
            f"Generation returned {len(generated.rcp_videos)} RCP videos, expected {len(view_plan.rcp_camera_ids)}."
        )
    if len(generated.target_videos) != view_plan.num_target_views:
        raise FourDAnyoneError(
            f"Generation returned {len(generated.target_videos)} target videos, expected {view_plan.num_target_views}."
        )
    if len(conditioning.target_skeletons) != view_plan.num_target_views:
        raise FourDAnyoneError(
            f"Conditioning returned {len(conditioning.target_skeletons)} target skeletons, "
            f"expected {view_plan.num_target_views}."
        )

    sparse_root = root / "videos" / "sparse"
    dense_root = root / "videos" / "dense"
    skeletons_root = root / "skeletons"
    dense_root.mkdir(parents=True, exist_ok=False)
    skeletons_root.mkdir(exist_ok=False)
    if generated.rcp_videos:
        sparse_root.mkdir(exist_ok=False)

    output_sparse = tuple(
        _copy_file(source, sparse_root / f"{camera_id:02d}.mp4")
        for camera_id, source in zip(view_plan.rcp_camera_ids, generated.rcp_videos, strict=True)
    )
    output_dense = tuple(
        _copy_file(source, dense_root / f"{camera_id:02d}.mp4")
        for camera_id, source in enumerate(generated.target_videos)
    )
    for camera_id, skeleton in enumerate(conditioning.target_skeletons):
        _copy_file(skeleton.path, skeletons_root / f"{camera_id:02d}.mp4")

    camera_payload = json.loads((conditioning.root / "cameras.json").read_text())
    conditioning_metadata = json.loads((conditioning.root / "metadata.json").read_text())
    camera_records = _target_cameras(camera_payload, view_plan.num_target_views)
    total_elapsed = time.monotonic() - pipeline_started
    generation_metadata = {
        "seed": generated.seed,
        "view_plan": {
            **view_plan.to_dict(),
            "num_layers": view_plan.num_layers,
            "num_target_views": view_plan.num_target_views,
            "groups_per_layer": view_plan.groups_per_layer,
        },
        "attention_backend": attention_backend,
        "inference_steps": 4 if generated.denoising_profile != "base24" else INFERENCE.num_inference_steps,
        "denoising_profile": generated.denoising_profile,
        "turbo_lora_sha256": generated.turbo_lora_sha256,
        "elapsed_seconds": generated.elapsed_seconds,
        "stage_peak_vram_bytes": generated.stage_peak_vram_bytes,
        "total_elapsed_seconds": total_elapsed,
        "peak_vram_allocated_bytes": generated.peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": generated.peak_vram_reserved_bytes,
        "peak_vram_process_bytes": generated.peak_vram_process_bytes,
        "execution_profile": generated.execution_profile,
    }
    if generated.parallelism is not None:
        generation_metadata["parallelism"] = generated.parallelism

    from fdanyone.audio import export_clip_audio
    audio = export_clip_audio(clip, root/'audio.m4a')

    metadata = {
        "input": {
            "filename": clip.source_path.name,
            "fps": f"{clip.fps_num}/{clip.fps_den}",
            "start_time_seconds": float(clip.start_time),
            "num_frames": len(clip.frames),
            "width": clip.width,
            "height": clip.height,
        },
        "motion": motion,
        "preprocessing": {
            "source_crop_policy": conditioning_metadata["source_crop_policy"],
            "foreground_model": conditioning_metadata["foreground_model"],
            "framing": conditioning_metadata["framing"],
            "skeleton_draw_scale": conditioning_metadata["skeleton_draw_scale"],
        },
        "model": dict(model_identity),
        "generation": generation_metadata,
        "output": {
            "rcp_views": len(output_sparse),
            "target_views": len(output_dense),
            "frames_per_video": INFERENCE.num_frames,
            "width": INFERENCE.width,
            "height": INFERENCE.height,
            "fps": f"{clip.fps_num}/{clip.fps_den}",
        },
        "runtime": _runtime_metadata(generated.device),
    }
    write_json(root / "cameras.json", _camera_rig_payload(camera_payload, camera_records))
    if audio is not None:
        metadata['audio'] = {'file': audio.name, 'mime_type': 'audio/mp4',
                             'duration': (len(clip.frames)-1)/float(clip.fps)}
    write_json(root / "metadata.json", metadata)
    return {
        "attention_backend": attention_backend,
        "num_rcp_videos": len(output_sparse),
        "num_target_videos": len(output_dense),
        "fps": f"{clip.fps_num}/{clip.fps_den}",
        "peak_vram_allocated_bytes": generated.peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": generated.peak_vram_reserved_bytes,
        "peak_vram_process_bytes": generated.peak_vram_process_bytes,
        "total_pipeline_elapsed_seconds": total_elapsed,
    }
