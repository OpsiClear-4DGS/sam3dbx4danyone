# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Direct, registry-free loading of the frozen Wan/SpaTem inference stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fdanyone.errors import AssetError, ConfigurationError

if TYPE_CHECKING:
    import torch
    from torch import nn

    from fdanyone.vendor.diffsynth.schedulers.flow_match import FlowMatchScheduler

POSE_ENCODER_PREFIX = "pose_encoder."


def compile_denoiser(model) -> None:
    """Compile attention chunks after moving/fusing weights, preserving state keys.

    CUDA graphs are deliberately disabled: view stages release their GPU models
    between calls. PyTorch's disk cache is reused by subsequent stream workers.
    """
    if getattr(model, '_fdanyone_compiled', False):
        return
    import logging
    import os
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
    import torch
    for block in model.blocks:
        for name in ('self_attn', 'self_attn_mvs', 'cross_attn'):
            attention = getattr(block, name)
            # Retain the outer one-view loop and its in-place output reuse.
            # Whole-block compilation hoists working buffers across chunks.
            attention._forward_batch = torch.compile(
                attention._forward_batch, dynamic=False,
                mode='max-autotune-no-cudagraphs')
    model._fdanyone_compiled = True
    logging.getLogger('fdanyone').info(
        'Bounded attention compilation enabled: %d BF16 blocks; first use includes compilation',
        len(model.blocks))


@dataclass(frozen=True)
class Denoiser:
    """The exact DiT, scheduler, and dtype used by one denoising process."""

    model: nn.Module
    scheduler: FlowMatchScheduler
    dtype: torch.dtype


_DENOISER_CACHE = None


def clear_denoiser_cache() -> None:
    """Discard the single process-local model after a failed queued job."""
    global _DENOISER_CACHE
    _DENOISER_CACHE = None


def _load_checkpoint(
    path: Path,
    *,
    include_prefix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise AssetError("safetensors is required to load the 4DAnyone checkpoint.") from exc
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = checkpoint.keys()
        keys = (
            key
            for key in checkpoint_keys
            if (include_prefix is None or key.startswith(include_prefix))
            and not any(key.startswith(prefix) for prefix in exclude_prefixes)
        )
        return {key: checkpoint.get_tensor(key) for key in keys}


def _strict_assign(module, state_dict: dict, label: str) -> None:
    """Load into a meta-initialized module without a second parameter copy."""

    try:
        incompatible = module.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError as exc:
        raise ConfigurationError("4DAnyone requires PyTorch >=2.8 for assign-based model loading.") from exc
    except RuntimeError as exc:
        raise AssetError(f"{label} is incompatible with the released architecture: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssetError(
            f"{label} strict load failed; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _load_dit(
    checkpoint_path: Path,
    *,
    ffn_chunk_views: int | None = None,
    attention_batch_limit: int | None = None,
):
    import torch

    from fdanyone.vendor.diffsynth.models.wan_video_dit import (
        MODEL_DIM,
        NUM_HEADS,
        FourDAnyoneDiT,
        precompute_freqs_cis_3d,
    )

    with torch.device("meta"):
        dit = FourDAnyoneDiT(
            ffn_chunk_views=ffn_chunk_views,
            attention_batch_limit=attention_batch_limit,
        )
    state_dict = _load_checkpoint(checkpoint_path, exclude_prefixes=(POSE_ENCODER_PREFIX,))
    _strict_assign(dit, state_dict, "4DAnyone DiT checkpoint")
    del state_dict
    # ``freqs`` is a derived, non-persistent tensor and therefore is not in the
    # state dict populated above.
    dit.freqs = precompute_freqs_cis_3d(MODEL_DIM // NUM_HEADS)
    return dit.eval().requires_grad_(False)


def load_pose_encoder(checkpoint_path: str | Path, device: str):
    """Load only the small pose encoder partition from the DiT checkpoint."""

    import torch

    from fdanyone.vendor.diffsynth.models.wan_video_dit import MODEL_DIM
    from fdanyone.vendor.diffsynth.models.wan_video_pose_encoder import PoseEncoder

    state_dict = {
        key.removeprefix(POSE_ENCODER_PREFIX): value
        for key, value in _load_checkpoint(Path(checkpoint_path), include_prefix=POSE_ENCODER_PREFIX).items()
    }
    with torch.device("meta"):
        pose_encoder = PoseEncoder(out_dim=MODEL_DIM, in_channels=3)
    _strict_assign(pose_encoder, state_dict, "4DAnyone pose encoder checkpoint")
    del state_dict
    return pose_encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def load_vae(path: str | Path):
    """Load BF16 encoding and FP16 decoding directly from the original weights."""
    import torch

    from fdanyone.vendor.diffsynth.models.wan_video_vae import WanVideoVAE38

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = WanVideoVAE38.state_dict_converter().from_civitai(state_dict)
    with torch.device("meta"):
        vae = WanVideoVAE38()
    _strict_assign(vae, state_dict, "Wan2.2 VAE")
    del state_dict
    # These tensors are derived attributes, not checkpoint entries. Recreate
    # them after strict assignment because construction happened on ``meta``.
    vae.materialize_normalization(device="cpu")
    # Casting the whole VAE to BF16 first would irreversibly round the decoder
    # weights before FP16 conversion. Keep the two stages independent.
    vae.model.encoder.to(dtype=torch.bfloat16)
    vae.model.conv1.to(dtype=torch.bfloat16)
    vae.model.decoder.to(dtype=torch.float16)
    vae.model.conv2.to(dtype=torch.float16)
    return vae.eval().requires_grad_(False)


def load_denoiser(
    *,
    checkpoint_path: str | Path,
    ffn_chunk_views: int | None = None,
    attention_batch_limit: int | None = None,
    cache_identity: object | None = None,
) -> Denoiser:
    """Load the DiT and scheduler required by one distributed worker."""

    import torch
    import os
    import logging

    from fdanyone.vendor.diffsynth.schedulers.flow_match import FlowMatchScheduler

    global _DENOISER_CACHE
    use_cache = os.environ.get('FDANYONE_REUSE_DENOISER') == '1' and cache_identity is not None
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, ffn_chunk_views,
           attention_batch_limit, cache_identity)
    if use_cache and _DENOISER_CACHE is not None and _DENOISER_CACHE[0] == key:
        logging.getLogger('fdanyone').info('Reusing loaded DiT and warmed compiled blocks')
        return _DENOISER_CACHE[1]
    # Bound host RAM to one cached denoiser per GPU worker.
    clear_denoiser_cache()
    dtype = torch.bfloat16
    result = Denoiser(
        model=_load_dit(
            Path(checkpoint_path),
            ffn_chunk_views=ffn_chunk_views,
            attention_batch_limit=attention_batch_limit,
        ),
        scheduler=FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True),
        dtype=dtype,
    )
    if use_cache:
        _DENOISER_CACHE = (key, result)
    return result
