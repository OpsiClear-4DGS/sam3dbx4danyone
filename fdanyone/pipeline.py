# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""SAM 3D Body + BiRefNet preprocessing and multiview generation."""
from __future__ import annotations
import logging
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from fdanyone.assets import (CHECKPOINT, HF_REPO_ID, HF_REVISION, SAM3D_REVISION,
    SAM3D_HF_REPO_ID, SAM3D_HF_REVISION, resolve_checkpoint, resolve_base_assets)
from fdanyone.config import INFERENCE, resolve_execution_profile
from fdanyone.device import CUDA_ALLOCATOR_CONF, select_cuda_devices
from fdanyone.download import ensure_example_video, ensure_models, ensure_sam3d, ensure_turbo, ensure_foreground_model
from fdanyone.errors import ConfigurationError
from fdanyone.io import AtomicResultDirectory, remove_tree, write_json, link_or_copy_file
from fdanyone.video import decode_canonical_clip, validate_required_video_codecs, verify_lossless_video, write_motion_video
from fdanyone.views import resolve_view_plan
LOGGER = logging.getLogger("fdanyone")

def _discard_scratch(path: Path) -> None:
    """Best-effort cleanup that can never invalidate a published result.

    Some network filesystems keep an open, hidden tombstone after a file is
    unlinked.  Such a tombstone may remain ``EBUSY`` until this process exits,
    so cleanup must not be part of the atomic publication transaction.
    """

    try:
        remove_tree(path)
    except OSError as exc:
        LOGGER.warning(
            "Could not remove temporary files at %s (%s). "
            "The result is unaffected; the hidden scratch directory can be removed after this process exits.",
            path,
            exc,
        )


def _worker_environment() -> dict[str, str]:
    """Give short-lived preprocessing workers this checkout and stable CUDA flags."""

    environment = os.environ.copy()
    environment.update(
        {
            "TORCH_CUDNN_V8_API_DISABLED": "1",
            "CUDNN_FRONTEND_DISABLE": "1",
            "CUDNN_LOGINFO_DBG": "0",
            "CUDNN_LOGDEST_DBG": "stderr",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    environment.pop("PYTHONHOME", None)
    # The 4 GiB split policy is specific to the long-lived DiT process. These
    # short-lived preprocessing workers use unrelated allocation shapes.
    environment.pop(CUDA_ALLOCATOR_CONF, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return environment


def run_pipeline(*, video_path, data_dir, model_dir, checkpoint_path, gpu_ids,
                 start_time, target_fps, seed, views_per_layer, layer_pitches,
                 start_yaw, yaw_span, views_per_group, enable_rcp, enable_tcr,
                 execution_profile, motion_backend="auto", motion_precision="fp16", turbo=False,
                 compile_dit=True, pose_dir=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.monotonic()
    if seed < 0:
        raise ConfigurationError("seed must be non-negative")
    execution = resolve_execution_profile(execution_profile)
    plan = resolve_view_plan(views_per_layer=views_per_layer, layer_pitches=layer_pitches,
        start_yaw=start_yaw, yaw_span=yaw_span, views_per_group=views_per_group,
        enable_rcp=enable_rcp and not turbo, enable_tcr=enable_tcr)
    if turbo:
        from fdanyone.model.inference import _turbo_target_routes
        _turbo_target_routes(plan)
    data_root = Path(data_dir).expanduser().resolve()
    destination = data_root / "fdanyone" / Path(video_path).stem
    atomic = AtomicResultDirectory(destination)
    if os.path.lexists(destination):
        raise ConfigurationError(f"Result already exists: {destination}. Choose a new --data_dir.")
    validate_required_video_codecs()
    devices = select_cuda_devices(gpu_ids)
    import torch
    from fdanyone.config import resolve_compilation_policy
    execution = resolve_compilation_policy(
        execution, enabled=compile_dit, turbo=turbo,
        total_memory_bytes=min(torch.cuda.get_device_properties(d).total_memory for d in devices),
        free_memory_bytes=min(torch.cuda.mem_get_info(d)[0] for d in devices))
    logging.getLogger('fdanyone').info(
        'DiT compilation: %s; process VRAM cap: %s GiB',
        execution.compile_dit,
        execution.process_vram_limit_bytes / 2**30 if execution.process_vram_limit_bytes else None)
    ensure_example_video(video_path)
    models = ensure_models(model_dir)
    source = ensure_sam3d(models)
    adapter = ensure_turbo(models) if turbo else None
    clip = decode_canonical_clip(video_path, num_frames=INFERENCE.num_frames,
        start_time=start_time, fps=None if str(target_fps).lower() == "auto" else target_fps)
    data_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".sam3d-", dir=data_root))
    try:
        clip_metadata = scratch / "canonical_clip.json"
        clip.write_metadata(clip_metadata)
        working = write_motion_video(clip, scratch / "canonical_clip.mp4")
        request = scratch / "request.json"
        preprocessing = scratch / "preprocessing"
        write_json(request, dict(working_video=str(working), clip_metadata=str(clip_metadata),
            output_dir=str(preprocessing), model_dir=str(models), source=str(source),
            device=devices[0], backend=motion_backend, precision=motion_precision,
            view_plan=plan.to_dict(), process_vram_limit_bytes=execution.process_vram_limit_bytes,
            saved_pose_dir=str(Path(pose_dir).resolve()) if pose_dir else None))
        subprocess.run([sys.executable, "-m", "fdanyone.motion.worker", str(request)],
            check=True, env=_worker_environment())
        from fdanyone.skeleton.pipeline import Conditioning
        from fdanyone.model.inference import generate_views
        from fdanyone.output import export_result
        motion_report = json.loads((preprocessing / "report.json").read_text())
        conditioning = Conditioning.load(preprocessing / "conditioning")
        verify_lossless_video(clip, conditioning.source_video)
        checkpoint = resolve_checkpoint(checkpoint_path, models)
        with atomic as work:
            generated = generate_views(clip=clip, conditioning=conditioning,
                checkpoint_path=checkpoint, assets=resolve_base_assets(models),
                output_dir=scratch / "generation", devices=devices, seed=seed,
                execution=execution, turbo_lora_path=adapter)
            report = export_result(clip=clip, conditioning=conditioning, generated=generated,
                destination=work, pipeline_started=started,
                model_identity=({"checkpoint": CHECKPOINT, "repo_id": HF_REPO_ID, "revision": HF_REVISION}
                    if checkpoint_path is None else {"checkpoint": checkpoint.name, "source": "local_override"}),
                motion={"method": "SAM 3D Body", "revision": SAM3D_REVISION,
                    "model_repo_id": SAM3D_HF_REPO_ID, "model_revision": SAM3D_HF_REVISION,
                    "body_model": "MHR", "box_and_mask_source": "BiRefNet",
                    "inference_backend": motion_report["inference_backend"],
                    "inference_precision": motion_report["inference_precision"],
                    "world_policy": "static-camera; one sequence rigid transform",
                    "limitations": ["Single person", "Static source camera assumed", "No temporal refinement"]})
            for name in ("sam3d_predictions.npz", "report.json", "geometry_checks.json", "viewer_geometry.npz"):
                link_or_copy_file(preprocessing / name, work / "preprocessing" / name)
        return {"result_dir": str(destination), **report}
    finally:
        _discard_scratch(scratch)


def extract_pose(*, video_path, output_dir, model_dir="models", target_fps="auto", precision="fp16", start_time=0):
    """Save camera-independent SAM pose and BiRefNet masks for later generation."""
    models = Path(model_dir).expanduser().resolve()
    ensure_foreground_model(models)
    source = ensure_sam3d(models)
    clip = decode_canonical_clip(video_path, num_frames=INFERENCE.num_frames,
        start_time=start_time, fps=None if str(target_fps).lower() == "auto" else target_fps)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pose-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        metadata = scratch / "canonical_clip.json"
        clip.write_metadata(metadata)
        working = write_motion_video(clip, scratch / "canonical_clip.mp4")
        with AtomicResultDirectory(output) as work:
            request = scratch / "request.json"
            write_json(request, dict(working_video=str(working), clip_metadata=str(metadata),
                output_dir=str(work), model_dir=str(models), source=str(source), device="cuda:0",
                backend="auto", precision=precision, pose_only=True,
                process_vram_limit_bytes=resolve_execution_profile("full").process_vram_limit_bytes))
            subprocess.run([sys.executable, "-m", "fdanyone.motion.worker", str(request)],
                check=True, env=_worker_environment())
    return {"pose_dir": str(output)}
