# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Fixed model and preprocessing settings used by the released method.

Only reader-useful choices live in the CLI. These values describe the trained
model and therefore stay together here instead of being exposed as knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from fdanyone.errors import ConfigurationError


@dataclass(frozen=True)
class InferenceConfig:
    num_frames: int = 121
    height: int = 1280
    width: int = 704
    prompt: str = "视频中的人在做动作"
    auto_downsample_fps: tuple[tuple[int, int], ...] = (
        (24, 1),
        (24000, 1001),
        (25, 1),
        (30, 1),
        (30000, 1001),
    )
    temporal_sampling_policy: str = "nearest_source_pts_on_zero_based_cfr_clock"
    rcp_jpeg_quality: int = 85
    skeleton_h264_crf: int = 17
    target_h264_crf: int = 18
    h264_preset: str = "medium"
    skeleton_max_dimension: int = 2048
    num_inference_steps: int = 24
    denoising_strength: float = 1.0
    scheduler_shift: float = 5.0
    vae_tile_size: tuple[int, int] = (52, 30)
    vae_tile_stride: tuple[int, int] = (26, 15)


GIB = 1 << 30


@dataclass(frozen=True)
class ExecutionProfile:
    """Inference precision, publication, and memory controls."""

    name: str
    pose_encoder_batch_limit: int
    pose_encoder_worker_limit: int | None
    pose_encoder_separate_nulls: bool
    dit_attention_batch_limit: int | None
    dit_ffn_chunk_views: int | None
    latent_accumulation_dtype: str
    vae_tile_size: tuple[int, int]
    vae_tile_stride: tuple[int, int]
    vae_decode_tile_size: tuple[int, int]
    vae_decode_tile_stride: tuple[int, int]
    lossless_target_video: bool
    target_video_preset: str
    cuda_max_split_size_mb: int
    cuda_expandable_segments: bool
    cuda_garbage_collection_threshold: float | None
    process_vram_target_bytes: int | None
    process_vram_limit_bytes: int | None

    compile_dit: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "compile_dit": self.compile_dit,
            "dit_compilation": 'attention_chunks' if self.compile_dit else 'eager',
            "pose_encoder_batch_limit": self.pose_encoder_batch_limit,
            "pose_encoder_worker_limit": self.pose_encoder_worker_limit,
            "pose_encoder_separate_nulls": self.pose_encoder_separate_nulls,
            "dit_attention_batch_limit": self.dit_attention_batch_limit,
            "dit_ffn_chunk_views": self.dit_ffn_chunk_views,
            "latent_accumulation_dtype": self.latent_accumulation_dtype,
            "tiled_vae": True,
            "vae_encoder_dtype": "bfloat16",
            "vae_decoder_dtype": "float16",
            "vae_rgb_conversion": "round_to_nearest_uint8",
            "vae_tile_size": list(self.vae_tile_size),
            "vae_tile_stride": list(self.vae_tile_stride),
            "vae_decode_tile_size": list(self.vae_decode_tile_size),
            "vae_decode_tile_stride": list(self.vae_decode_tile_stride),
            "lossless_target_video": self.lossless_target_video,
            "target_video_preset": self.target_video_preset,
            "cuda_max_split_size_mb": self.cuda_max_split_size_mb,
            "cuda_expandable_segments": self.cuda_expandable_segments,
            "cuda_garbage_collection_threshold": self.cuda_garbage_collection_threshold,
            "process_vram_target_bytes": self.process_vram_target_bytes,
            "process_vram_limit_bytes": self.process_vram_limit_bytes,
        }


@dataclass(frozen=True)
class CameraConfig:
    count: int = 24
    pitch_degrees: float = 15.0

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("Camera count must be positive.")


@dataclass(frozen=True)
class ForegroundConfig:
    """Pinned standard BiRefNet inference contract."""

    image_size: tuple[int, int] = (1024, 1024)
    batch_size: int = 4


@dataclass(frozen=True)
class FramingConfig:
    """Sequence-level camera solve for static-camera motion conditioning."""

    reference_radius: float = 3.0
    reference_target_height: float = 1.0
    reference_focal_normalized: float = 1664.0 / 1280.0
    height_target_ratio: float = 0.80
    height_percentile: float = 95.0
    width_target_ratio: float = 0.90
    width_percentile: float = 80.0
    min_radius: float = 1.5
    max_radius: float = 8.0
    input_min_confidence: float = 0.55
    max_focal_normalized: float = 4.0
    cutoff_target_ratio: float = 0.99
    cutoff_percentile: float = 80.0


@dataclass(frozen=True)
class CropConfig:
    """Source-mask crop; generated cameras use a plain center aspect crop."""

    margin_top: float = 0.04
    margin_right: float = 0.04
    margin_bottom: float = 0.04
    margin_left: float = 0.04
    allow_upscale: bool = True
    mask_threshold: float = 0.05

    @property
    def margins(self) -> tuple[float, float, float, float]:
        return (
            self.margin_top,
            self.margin_right,
            self.margin_bottom,
            self.margin_left,
        )


@dataclass(frozen=True)
class SkeletonConfig:
    draw_body_reference_px: float = 640.0


INFERENCE = InferenceConfig()
FULL_EXECUTION = ExecutionProfile(
    name="full",
    pose_encoder_batch_limit=1,
    pose_encoder_worker_limit=1,
    pose_encoder_separate_nulls=True,
    dit_attention_batch_limit=1,
    dit_ffn_chunk_views=1,
    # The BF16 DiT still sees the checkpoint's native input dtype. Only the
    # Euler state is kept in FP32 so rounding error does not accumulate across
    # all 24 scheduler updates.
    latent_accumulation_dtype="float32",
    vae_tile_size=INFERENCE.vae_tile_size,
    vae_tile_stride=INFERENCE.vae_tile_stride,
    # Preserve the decoder's final uint8 RGB values instead of introducing a
    # second lossy H.264 boundary before reconstruction or review.
    vae_decode_tile_size=(52, 44),
    vae_decode_tile_stride=(28, 44),
    lossless_target_video=True,
    # Lossless RGB values are preset-independent. ``veryfast`` measured 32.9%
    # faster than ``medium`` with 2.1% more bytes across six production views.
    target_video_preset="veryfast",
    cuda_max_split_size_mb=4096,
    cuda_expandable_segments=True,
    cuda_garbage_collection_threshold=0.70,
    process_vram_target_bytes=int(22.5 * GIB),
    process_vram_limit_bytes=23 * GIB,
)
CAMERA = CameraConfig()
FOREGROUND = ForegroundConfig()
FRAMING = FramingConfig()
CROP = CropConfig()
SKELETON = SkeletonConfig()


def resolve_compilation_policy(execution: ExecutionProfile, *, enabled: bool,
                               turbo: bool, total_memory_bytes: int,
                               free_memory_bytes: int) -> ExecutionProfile:
    """Compile inside attention chunks without relaxing the strict VRAM cap."""
    compile_dit = bool(enabled and turbo and execution.name == 'full'
                       and total_memory_bytes >= 23 * GIB
                       and free_memory_bytes >= 23 * GIB)
    return replace(execution, compile_dit=compile_dit)


def resolve_execution_profile(value: str | ExecutionProfile) -> ExecutionProfile:
    """Resolve a stable public execution-profile name."""

    if isinstance(value, ExecutionProfile):
        return value
    if not isinstance(value, str):
        raise ConfigurationError(f"execution_profile must be a string, got {value!r}.")
    normalized = value.strip().lower().replace("_", "-")
    profiles = {
        FULL_EXECUTION.name: FULL_EXECUTION,
        "quality": FULL_EXECUTION,  # Compatibility with existing commands.
    }
    try:
        return profiles[normalized]
    except KeyError:
        raise ConfigurationError(
            f"execution_profile must be one of {sorted(profiles)}, got {value!r}."
        ) from None
