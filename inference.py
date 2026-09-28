# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Run 4DAnyone inference from a monocular video."""

from __future__ import annotations

import sys

from fdanyone.device import configure_inference_cuda_allocator, configure_inference_cpu_environment
from fdanyone.errors import FourDAnyoneError


def inference(
    video_path: str,
    views_per_layer: int = 6,
    layer_pitches: list[int] = [15],  # noqa: B006 - normalized without mutation
    start_yaw: int = 0,
    yaw_span: int = 360,
    views_per_group: int | str = "auto",
    enable_rcp: bool = True,
    enable_tcr: bool = True,
    data_dir: str = "data",
    model_dir: str = "models",
    checkpoint_path: str | None = None,
    gpu_ids: list[int] | None = None,
    target_fps: str | int | float = "auto",
    start_time: float = 0.0,
    seed: int = 42,
    execution_profile: str = "full",
    motion_backend: str = "auto",
    motion_precision: str = "fp16",
    turbo: bool = True,
    compile_dit: bool = True,
    train_4dgs: bool = False,
    training_steps: int = 30_000,
) -> dict:
    """Generate synchronized target-view videos from one monocular video.

    Args:
        video_path: Input video; it must contain at least 121 usable frames.
        views_per_layer: Number of evenly spaced yaw views at each pitch. It
            must be divisible by 4 or 6.
        layer_pitches: Camera pitch for each layer in degrees, for example
            [-10,15,35]. Positive values place the camera above the subject;
            each value must be between -15 and 45.
        start_yaw: First yaw in every layer, in degrees; 0 faces the person.
        yaw_span: Angular range sampled by each layer, from 1 to 360 degrees.
            The end angle is excluded so a full ring never duplicates a view.
        views_per_group: Maximum target views generated together. auto chooses
            6 when possible and otherwise 4; a manual value must be 4 or 6 and
            divide views_per_layer.
        enable_rcp: Use proposal views before generating more than six targets.
            The proposal count follows views_per_group.
        enable_tcr: Shift view groups cyclically between denoising steps.
        data_dir: Root for SAM 3D Body preprocessing and final 4DAnyone outputs.
        model_dir: Model root; missing public checkpoints download here.
        checkpoint_path: Local 4DAnyone checkpoint override.
        gpu_ids: GPU IDs used for parallel pose/VAE view stages and target
            denoising. Omit to use all visible GPUs.
        target_fps: auto preserves the input clock unless it divides evenly
            to 24, 25, or 30 FPS; a positive number requests an explicit FPS.
        start_time: Clip start time on the input timeline, in seconds.
        seed: Random seed shared by proposal and target generation.
        execution_profile: ``full`` (default) keeps the full VAE, FP32
            denoising state and lossless targets under the 23 GiB process cap.
            ``quality`` is a compatibility alias for the same mode.
        motion_backend: SAM backbone execution: auto, tensorrt, or cuda-ort.
        motion_precision: TensorRT precision: fp16 (validated fast path) or fp32.
        turbo: Four-step generation, enabled by default; supports 6 or experimental 24 views in one layer; uses six-view groups without RCP.
        compile_dit: Compile bounded Turbo attention chunks in the full
            profile when at least 23 GiB is free. The VRAM cap remains 23 GiB.
            False selects the eager reference; queued videos reuse warmed code.
        train_4dgs: Train a foreground FreeTimeGS scene after video generation.
        training_steps: Optimizer steps for each independent 4DGS chunk.
    """

    from fdanyone.config import resolve_execution_profile
    from fdanyone.trt import normalize_motion_backend, normalize_trt_precision

    if train_4dgs:
        from fdanyone.reconstruction import check_runtime
        from fdanyone.errors import ConfigurationError
        if isinstance(training_steps, bool) or not isinstance(training_steps, int) or training_steps < 1:
            raise ConfigurationError('training_steps must be a positive integer.')
        check_runtime(probe=True)

    configure_inference_cpu_environment()
    execution = resolve_execution_profile(execution_profile)
    motion_backend = normalize_motion_backend(motion_backend)
    motion_precision = normalize_trt_precision(motion_precision)
    # This must run before the first model/PyTorch import. The bounded profiles
    # use expandable segments and proactive cache reclamation to keep the
    # process footprint close to their live tensor allocation.
    configure_inference_cuda_allocator(
        max_split_size_mb=execution.cuda_max_split_size_mb,
        expandable_segments=execution.cuda_expandable_segments,
        garbage_collection_threshold=execution.cuda_garbage_collection_threshold,
    )
    # Keep model imports out of module scope so ``--help`` stays lightweight.
    from fdanyone.pipeline import run_pipeline

    result = run_pipeline(
        video_path=video_path,
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        views_per_group=views_per_group,
        enable_rcp=enable_rcp,
        enable_tcr=enable_tcr,
        data_dir=data_dir,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        gpu_ids=gpu_ids,
        target_fps=target_fps,
        start_time=start_time,
        seed=seed,
        execution_profile=execution.name,
        motion_backend=motion_backend,
        motion_precision=motion_precision,
        turbo=turbo,
        compile_dit=compile_dit,
    )
    if train_4dgs:
        from fdanyone.reconstruction import train_result
        result['training'] = train_result(result['result_dir'], model_dir=model_dir,
            gpu_id=gpu_ids[0] if gpu_ids else 0, steps=training_steps)
        if 'total_pipeline_elapsed_seconds' in result:
            result['generation_pipeline_elapsed_seconds'] = result['total_pipeline_elapsed_seconds']
            result['total_pipeline_elapsed_seconds'] += result['training']['elapsed_seconds']
    return result


def main() -> None:
    """Bootstrap the CLI without importing Fire or PyTorch at module import."""

    from fire import Fire

    try:
        Fire(inference)
    except FourDAnyoneError as exc:
        message = " ".join(line.strip() for line in str(exc).splitlines())
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
