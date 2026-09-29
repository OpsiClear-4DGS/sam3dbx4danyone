# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Hydra/Lightning-free multi-view generation.

RCP and final target generation share one source encoding and prompt embedding.
Target groups execute sequentially on one GPU or concurrently across multiple
GPUs; TCR optionally shifts their membership between denoising steps.
"""

from __future__ import annotations

import gc
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, TypedDict

from fdanyone.assets import BaseAssets
from fdanyone.config import INFERENCE, ExecutionProfile
from fdanyone.errors import FourDAnyoneError
from fdanyone.model.conditioning import (
    PoseFeatureBank,
    PoseFeatureCache,
    build_pose_feature_cache,
    load_prompt_context,
)
from fdanyone.model.denoise import denoise_group
from fdanyone.model.distributed import (
    Routes,
    WorkerReport,
    denoise_targets_distributed,
    select_worker_devices,
)
from fdanyone.model.loader import Denoiser, load_denoiser
from fdanyone.model.metrics import GenerationMetrics
from fdanyone.model.routing import routing_steps
from fdanyone.model.vae import VaeExecutor, load_reference_videos
from fdanyone.skeleton.pipeline import Conditioning
from fdanyone.video import CanonicalClip
from fdanyone.views import ViewPlan

if TYPE_CHECKING:
    from torch import Tensor, nn

LOGGER = logging.getLogger("fdanyone")


class ParallelismReport(TypedDict):
    """The target-denoising topology and measurements written to metadata."""

    backend: str
    candidate_devices: list[str]
    used_devices: list[str]
    groups_per_step: int
    waves_per_step: int
    workers: list[WorkerReport]


@dataclass(frozen=True)
class GeneratedViews:
    """Paths and measurements produced by one resolved view plan."""

    rcp_videos: tuple[Path, ...]
    target_videos: tuple[Path, ...]
    view_plan: ViewPlan
    seed: int
    device: str
    elapsed_seconds: dict[str, float]
    stage_peak_vram_bytes: dict[str, dict[str, int]]
    peak_vram_allocated_bytes: int
    peak_vram_reserved_bytes: int
    peak_vram_process_bytes: int
    execution_profile: dict[str, object]
    parallelism: ParallelismReport | None = None
    denoising_profile: str = "base24"
    turbo_lora_sha256: str | None = None


@dataclass(frozen=True)
class _GenerationPlan:
    view_plan: ViewPlan
    candidate_devices: tuple[str, ...]
    primary_device: str
    primary_device_index: int
    dit_devices: tuple[str, ...]

    @property
    def view_devices(self) -> tuple[str, ...]:
        return self.candidate_devices

    @property
    def distributed(self) -> bool:
        return len(self.dit_devices) > 1

    @property
    def needs_primary_denoiser(self) -> bool:
        return self.view_plan.enable_rcp or not self.distributed


def _empty_cuda_cache() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _model_on_device(model: nn.Module, device: str) -> Iterator[nn.Module]:
    """Keep one model resident for exactly one exception-safe stage."""

    model.to(device)
    _empty_cuda_cache()
    try:
        yield model
    finally:
        model.to("cpu")
        _empty_cuda_cache()


def _bf16_autocast():
    """Match Lightning's ``bf16-mixed`` inference context without Lightning."""

    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _channels_last_source_layout(video):
    """Preserve the frozen source tensor's VFHWC-backed VCFHW layout."""

    import torch

    if video.ndim != 5:
        raise FourDAnyoneError(
            f"Expected a 5D video tensor, got shape {tuple(video.shape)}."
        )
    return video.contiguous(memory_format=torch.channels_last_3d)


def _target_decode_devices(
    devices: tuple[str, ...], primary_device: str
) -> tuple[str, ...]:
    """Avoid reusing the long-lived DiT context when another GPU is available."""

    alternatives = tuple(device for device in devices if device != primary_device)
    return alternatives or devices


def _latent_dtype(name: str):
    """Resolve the two supported scheduler-state dtypes without importing torch in config."""

    import torch

    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return dtypes[name]
    except KeyError:
        raise FourDAnyoneError(
            f"Unsupported latent accumulation dtype {name!r}; expected one of {sorted(dtypes)}."
        ) from None


def _noise(
    vae: VaeExecutor,
    num_views: int,
    num_frames: int,
    seed: int,
    device: str,
    dtype,
) -> Tensor:
    import torch

    shape = (
        num_views,
        vae.latent_channels,
        (num_frames - 1) // 4 + 1,
        INFERENCE.height // vae.upsampling_factor,
        INFERENCE.width // vae.upsampling_factor,
    )
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(
        shape, generator=generator, device="cpu", dtype=torch.float32
    ).to(dtype=dtype, device=device)


def _denoise_rcp(
    denoiser: Denoiser,
    vae: VaeExecutor,
    src_latents: Tensor,
    context: Tensor,
    camera_ids: tuple[int, ...],
    pose_features: PoseFeatureBank,
    device: str,
    seed: int,
    latent_dtype,
) -> Tensor:
    import torch
    from tqdm.auto import tqdm

    if pose_features.num_features != len(camera_ids):
        raise FourDAnyoneError(
            f"RCP requires {len(camera_ids)} pose features, got {pose_features.num_features}."
        )
    denoiser.scheduler.set_timesteps(
        INFERENCE.num_inference_steps,
        denoising_strength=INFERENCE.denoising_strength,
        shift=INFERENCE.scheduler_shift,
    )
    latents = _noise(
        vae,
        len(camera_ids),
        INFERENCE.num_frames,
        seed,
        device,
        latent_dtype,
    )
    source = src_latents.to(dtype=denoiser.dtype, device=device)
    context = context.to(dtype=denoiser.dtype, device=device)
    pose_feature_batch = pose_features.allocate_group(len(camera_ids), device)
    pose_features.copy_group(tuple(range(len(camera_ids))), pose_feature_batch)
    null_pose_feature = pose_features.null_on(device)

    with torch.inference_mode(), _bf16_autocast():
        for step_index, _ in enumerate(
            tqdm(denoiser.scheduler.timesteps, desc=f"RCP 1-to-{len(camera_ids)}")
        ):
            latents = denoise_group(
                denoiser,
                latents,
                source,
                context,
                pose_feature_batch,
                null_pose_feature,
                step_index,
            )
    return latents.detach().to("cpu")


def _denoise_targets_single(
    denoiser: Denoiser,
    src_latents: Tensor,
    context: Tensor,
    pose_features: PoseFeatureBank,
    initial_latents: Tensor,
    routes: Routes,
    device: str,
) -> Tensor:
    import torch
    from tqdm.auto import tqdm

    num_views = initial_latents.shape[0]
    if pose_features.num_features != num_views:
        raise FourDAnyoneError(
            f"Target generation requires {num_views} pose features, got {pose_features.num_features}."
        )
    denoiser.scheduler.set_timesteps(
        len(routes),
        denoising_strength=INFERENCE.denoising_strength,
        shift=INFERENCE.scheduler_shift,
    )
    latents = initial_latents
    source = src_latents.to(dtype=denoiser.dtype, device=device)
    context = context.to(dtype=denoiser.dtype, device=device)
    null_pose_feature = pose_features.null_on(device)
    group_size = len(routes[0][0])
    pose_feature_batch = pose_features.allocate_group(group_size, device)

    with torch.inference_mode(), _bf16_autocast():
        for step_index, groups in enumerate(
            tqdm(routes, desc=f"Generate {num_views} target views")
        ):
            for view_indices in groups:
                index = torch.tensor(view_indices, dtype=torch.long, device=device)
                local_latents = torch.index_select(latents, 0, index)
                pose_features.copy_group(view_indices, pose_feature_batch)
                local_latents = denoise_group(
                    denoiser,
                    local_latents,
                    source,
                    context,
                    pose_feature_batch,
                    null_pose_feature,
                    step_index,
                )
                latents.index_copy_(0, index, local_latents)
                del local_latents
    return latents.detach().to("cpu")


def _resolve_generation_plan(
    clip: CanonicalClip,
    conditioning: Conditioning,
    devices: tuple[str, ...],
    seed: int,
) -> _GenerationPlan:
    import torch

    if (
        conditioning.num_frames != INFERENCE.num_frames
        or len(clip.frames) != INFERENCE.num_frames
    ):
        raise FourDAnyoneError("Generation requires the frozen 121-frame contract.")
    if seed < 0:
        raise FourDAnyoneError(f"seed must be non-negative, got {seed}.")
    if not devices:
        raise FourDAnyoneError("Generation requires at least one CUDA device.")

    view_plan = conditioning.view_plan
    if len(conditioning.target_skeletons) != view_plan.num_target_views:
        raise FourDAnyoneError(
            "Target skeleton count does not match the resolved view plan."
        )
    if len(conditioning.rcp_skeletons) != len(view_plan.rcp_camera_ids):
        raise FourDAnyoneError(
            "RCP skeleton count does not match the resolved view plan."
        )

    from fdanyone.vendor.diffsynth.models.wan_video_dit import get_attention_backend
    backend = get_attention_backend()
    LOGGER.info("Attention backend: %s", backend)
    if backend == "sdpa":
        LOGGER.warning("SageAttention is unavailable; using slower SDPA attention.")
    primary_device = devices[0]
    primary_device_index = int(primary_device.removeprefix("cuda:"))
    dit_devices = select_worker_devices(devices, view_plan.num_groups)
    LOGGER.info(
        "Using %s (%s)",
        primary_device,
        torch.cuda.get_device_name(primary_device_index),
    )
    if len(dit_devices) < len(devices):
        LOGGER.info(
            "Using %d of %d candidate GPUs for DiT and all %d for independent view stages",
            len(dit_devices),
            len(devices),
            len(devices),
        )
    return _GenerationPlan(
        view_plan=view_plan,
        candidate_devices=devices,
        primary_device=primary_device,
        primary_device_index=primary_device_index,
        dit_devices=dit_devices,
    )


def _generate_rcp_and_references(
    *,
    denoiser: Denoiser,
    vae: VaeExecutor,
    source_latents: Tensor,
    context: Tensor,
    pose_cache: PoseFeatureCache,
    plan: _GenerationPlan,
    root: Path,
    clip: CanonicalClip,
    seed: int,
    latent_dtype,
    metrics: GenerationMetrics,
) -> tuple[Tensor, tuple[Path, ...]]:
    import torch

    rcp_pose_features = pose_cache.rcp
    if rcp_pose_features is None:
        raise FourDAnyoneError(
            "RCP was enabled without precomputed proposal pose features."
        )
    with metrics.stage("rcp_denoise"), _model_on_device(
        denoiser.model, plan.primary_device
    ):
        rcp_latents = _denoise_rcp(
            denoiser,
            vae,
            source_latents,
            context,
            plan.view_plan.rcp_camera_ids,
            rcp_pose_features,
            plan.primary_device,
            seed,
            latent_dtype,
        )

    with metrics.stage("rcp_decode_and_publish"):
        rcp_root = root / "rcp"
        rcp_root.mkdir()
        published = vae.publish_rcp(
            rcp_latents,
            plan.view_plan.rcp_camera_ids,
            rcp_root,
            clip,
        )
    _merge_view_stage_peak(metrics, "rcp_decode_and_publish", vae)

    with metrics.stage("rcp_reference_load"):
        # The released model consumes four JPEG-decoded proposal views. Keep
        # that numerical boundary even though the decoded tensors are local.
        reference_videos = load_reference_videos(
            published.frame_directories[:4], INFERENCE.num_frames
        )

    with metrics.stage("reference_encode"):
        reference_latents = vae.encode(reference_videos)
        target_sources = torch.cat([source_latents, reference_latents], dim=0)
    _merge_view_stage_peak(metrics, "reference_encode", vae)
    return target_sources, published.videos


def _merge_view_stage_peak(
    metrics: GenerationMetrics, stage: str, vae: VaeExecutor
) -> None:
    if not vae.last_peak_vram_bytes:
        return
    metrics.merge_cuda_peak(
        stage,
        allocated_bytes=max(
            peak["allocated"] for peak in vae.last_peak_vram_bytes.values()
        ),
        reserved_bytes=max(
            peak["reserved"] for peak in vae.last_peak_vram_bytes.values()
        ),
        process_bytes=max(
            peak["process"] for peak in vae.last_peak_vram_bytes.values()
        ),
    )
    metrics.enforce_process_limit(stage)


def _target_routes(view_plan: ViewPlan) -> Routes:
    return routing_steps(
        views_per_layer=view_plan.views_per_layer,
        num_layers=view_plan.num_layers,
        group_size=view_plan.views_per_group,
        num_steps=INFERENCE.num_inference_steps,
        enable_tcr=view_plan.enable_tcr,
    )


def _turbo_target_routes(view_plan: ViewPlan) -> Routes:
    """Four-step routing; extra cameras, rings and four-view groups are experimental."""
    if (view_plan.num_layers > 3
            or view_plan.num_target_views not in (6, 8, 12, 16, 18, 24, 36)
            or view_plan.enable_rcp):
        raise FourDAnyoneError("Turbo supports 6, 8, 12, 16, 18, 24 or 36 cameras across up to three rings, without RCP.")
    offsets = (0, 2, 4, 0) if view_plan.enable_tcr else (0, 0, 0, 0)
    # Reuse layer-local routing where possible; 9 x 2 uses cross-ring groups.
    candidates = routing_steps(
        views_per_layer=view_plan.views_per_layer,
        num_layers=view_plan.num_layers,
        group_size=view_plan.views_per_group,
        num_steps=5,
        enable_tcr=True,
    )
    return tuple(candidates[offset] for offset in offsets)


def _denoise_targets_multi_gpu(
    *,
    target_sources: Tensor,
    context: Tensor,
    initial_latents: Tensor,
    pose_features: PoseFeatureBank,
    routes: Routes,
    checkpoint_path: str | Path,
    root: Path,
    devices: tuple[str, ...],
    candidate_devices: tuple[str, ...],
    num_groups: int,
    ffn_chunk_views: int | None,
    attention_batch_limit: int | None,
    process_vram_limit_bytes: int | None,
    turbo_lora_path: Path | None = None,
    compile_dit: bool = False,
) -> tuple[Tensor, ParallelismReport]:
    with TemporaryDirectory(prefix=".distributed-", dir=root) as work_dir:
        target_latents, workers = denoise_targets_distributed(
            src_latents=target_sources,
            context=context,
            initial_latents=initial_latents,
            pose_features=pose_features,
            routes=routes,
            checkpoint_path=checkpoint_path,
            work_dir=work_dir,
            devices=devices,
            denoising_strength=INFERENCE.denoising_strength,
            scheduler_shift=INFERENCE.scheduler_shift,
            ffn_chunk_views=ffn_chunk_views,
            attention_batch_limit=attention_batch_limit,
            process_vram_limit_bytes=process_vram_limit_bytes,
            turbo_lora_path=turbo_lora_path,
            compile_dit=compile_dit,
        )
    return target_latents, {
        "backend": "nccl",
        "candidate_devices": list(candidate_devices),
        "used_devices": list(devices),
        "groups_per_step": num_groups,
        "waves_per_step": math.ceil(num_groups / len(devices)),
        "workers": workers,
    }


def generate_views(
    *,
    clip: CanonicalClip,
    conditioning: Conditioning,
    checkpoint_path: str | Path,
    assets: BaseAssets,
    output_dir: str | Path,
    devices: tuple[str, ...],
    seed: int,
    execution: ExecutionProfile,
    turbo_lora_path: str | Path | None = None,
) -> GeneratedViews:
    """Generate the proposal (when enabled) and the requested target views."""

    plan = _resolve_generation_plan(clip, conditioning, devices, seed)
    turbo_path = None
    turbo_sha256 = None
    if turbo_lora_path is not None:
        from safetensors import safe_open
        from fdanyone.model.turbo_lora import (
            TURBO_SHA256, validate_turbo_adapter, validate_turbo_base_metadata,
        )
        _turbo_target_routes(plan.view_plan)
        turbo_path = validate_turbo_adapter(turbo_lora_path)
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            validate_turbo_base_metadata(checkpoint.metadata())
        turbo_sha256 = TURBO_SHA256
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    metrics = GenerationMetrics(
        plan.primary_device_index,
        process_vram_limit_bytes=execution.process_vram_limit_bytes,
    )
    latent_dtype = _latent_dtype(execution.latent_accumulation_dtype)

    with metrics.stage("prompt"):
        context = load_prompt_context(assets.prompt_context)

    with metrics.stage("pose_conditioning"):
        pose_cache = build_pose_feature_cache(
            conditioning=conditioning,
            checkpoint_path=checkpoint_path,
            devices=plan.view_devices,
            batch_limit=execution.pose_encoder_batch_limit,
            worker_limit=execution.pose_encoder_worker_limit,
            separate_nulls=execution.pose_encoder_separate_nulls,
            process_vram_limit_bytes=execution.process_vram_limit_bytes,
        )
    if pose_cache.worker_peak_vram_bytes:
        metrics.merge_cuda_peak(
            "pose_conditioning",
            allocated_bytes=max(
                peak["allocated"] for peak in pose_cache.worker_peak_vram_bytes.values()
            ),
            reserved_bytes=max(
                peak["reserved"] for peak in pose_cache.worker_peak_vram_bytes.values()
            ),
            process_bytes=max(
                peak["process"] for peak in pose_cache.worker_peak_vram_bytes.values()
            ),
        )
        metrics.enforce_process_limit("pose_conditioning")

    with metrics.stage("model_load"):
        denoiser = (
            load_denoiser(
                checkpoint_path=checkpoint_path,
                ffn_chunk_views=execution.dit_ffn_chunk_views,
                attention_batch_limit=execution.dit_attention_batch_limit,
                cache_identity=(str(turbo_path), turbo_sha256,
                                turbo_path.stat().st_mtime_ns if turbo_path else None,
                                execution.compile_dit),
            )
            if plan.needs_primary_denoiser
            else None
        )
        vae = VaeExecutor.load(assets.vae, plan.view_devices, execution)

    try:
        with metrics.stage("source_encode"):
            source_video = _channels_last_source_layout(
                conditioning.load_source_tensor()
            )
            source_latents = vae.encode(source_video)
            del source_video
        _merge_view_stage_peak(metrics, "source_encode", vae)

        rcp_videos: tuple[Path, ...] = ()
        target_sources = source_latents
        if plan.view_plan.enable_rcp:
            if denoiser is None:
                raise RuntimeError("RCP requires a primary-process denoiser.")
            target_sources, rcp_videos = _generate_rcp_and_references(
                denoiser=denoiser,
                vae=vae,
                source_latents=source_latents,
                context=context,
                pose_cache=pose_cache,
                plan=plan,
                root=root,
                clip=clip,
                seed=seed,
                latent_dtype=latent_dtype,
                metrics=metrics,
            )
        del source_latents
        # Parallel VAE replicas are stage-local. Retaining only the prototype
        # bounds parent host memory while distributed DiT workers load.
        vae.release_replicas()
        target_pose_features = pose_cache.target
        del pose_cache

        parallelism = None
        with metrics.stage("target_denoise"):
            routes = (_target_routes(plan.view_plan) if turbo_path is None
                      else _turbo_target_routes(plan.view_plan))
            initial_latents = _noise(
                vae,
                plan.view_plan.num_target_views,
                INFERENCE.num_frames,
                seed,
                plan.primary_device,
                latent_dtype,
            )
            if plan.distributed:
                denoiser = None
                initial_latents = initial_latents.to("cpu")
                _empty_cuda_cache()
                target_latents, parallelism = _denoise_targets_multi_gpu(
                    target_sources=target_sources,
                    context=context,
                    initial_latents=initial_latents,
                    pose_features=target_pose_features,
                    routes=routes,
                    checkpoint_path=checkpoint_path,
                    root=root,
                    devices=plan.dit_devices,
                    candidate_devices=plan.candidate_devices,
                    num_groups=plan.view_plan.num_groups,
                    ffn_chunk_views=execution.dit_ffn_chunk_views,
                    attention_batch_limit=execution.dit_attention_batch_limit,
                    process_vram_limit_bytes=execution.process_vram_limit_bytes,
                    turbo_lora_path=turbo_path,
                    compile_dit=execution.compile_dit,
                )
            else:
                if denoiser is None:
                    raise RuntimeError(
                        "Single-GPU target generation requires a primary-process denoiser."
                    )
                with _model_on_device(denoiser.model, plan.primary_device):
                    if turbo_path is not None:
                        from fdanyone.model.turbo_lora import fuse_turbo_lora
                        fuse_turbo_lora(denoiser.model, turbo_path)
                        _empty_cuda_cache()
                    if execution.compile_dit:
                        from fdanyone.model.loader import compile_denoiser
                        compile_denoiser(denoiser.model)
                    target_latents = _denoise_targets_single(
                        denoiser,
                        target_sources,
                        context,
                        target_pose_features,
                        initial_latents,
                        routes,
                        plan.primary_device,
                    )
                denoiser = None
            del initial_latents

        if parallelism is not None:
            workers = parallelism["workers"]
            metrics.merge_cuda_peak(
                "target_denoise",
                allocated_bytes=max(
                    int(worker["peak_vram_allocated_bytes"]) for worker in workers
                ),
                reserved_bytes=max(
                    int(worker["peak_vram_reserved_bytes"]) for worker in workers
                ),
                process_bytes=max(
                    int(worker["peak_vram_process_bytes"]) for worker in workers
                ),
            )
            metrics.enforce_process_limit("target_denoise")
        del target_pose_features, target_sources, context

        with metrics.stage("target_decode_and_publish"):
            decode_devices = _target_decode_devices(vae.devices, plan.primary_device)
            if decode_devices != vae.devices:
                LOGGER.info(
                    "Reserving %s after DiT; decoding target views on %s",
                    plan.primary_device,
                    ", ".join(decode_devices),
                )
                vae.devices = decode_devices
            target_root = root / "target"
            target_root.mkdir()
            # Fresh processes release encoder/DiT CUDA workspaces before decode.
            from fdanyone.model.decode_worker import publish_isolated
            vae.close()
            _empty_cuda_cache()
            LOGGER.info("Decoding full FP16 VAE in fresh processes on %s", ", ".join(decode_devices))
            target_videos, vae.last_peak_vram_bytes = publish_isolated(
                target_latents, assets.vae, target_root, clip.fps, decode_devices, execution)
        _merge_view_stage_peak(metrics, "target_decode_and_publish", vae)

        result = GeneratedViews(
            rcp_videos=rcp_videos,
            target_videos=target_videos,
            view_plan=plan.view_plan,
            seed=seed,
            device=plan.primary_device,
            elapsed_seconds=metrics.elapsed_seconds,
            stage_peak_vram_bytes=metrics.stage_peak_vram_bytes,
            peak_vram_allocated_bytes=metrics.peak_vram_allocated_bytes,
            peak_vram_reserved_bytes=metrics.peak_vram_reserved_bytes,
            peak_vram_process_bytes=metrics.peak_vram_process_bytes,
            execution_profile=execution.to_dict(),
            parallelism=parallelism,
            denoising_profile="base24" if turbo_path is None else "rank64_delta4",
            turbo_lora_sha256=turbo_sha256,
        )
    finally:
        vae.close()
        _empty_cuda_cache()

    return result
