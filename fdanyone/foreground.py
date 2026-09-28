# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Pinned BiRefNet inference over the canonical source clip."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from fdanyone.config import FOREGROUND


def predict_foreground_masks(
    frames: tuple[np.ndarray, ...],
    model_path: str | Path,
    device: str,
    *,
    batch_size: int = FOREGROUND.batch_size,
) -> np.ndarray:
    """Return full-raster 8-bit foreground masks for the canonical clip."""

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageSegmentation

    if not frames:
        raise ValueError("Foreground inference requires at least one frame.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    shape = frames[0].shape
    if any(frame.dtype != np.uint8 or frame.shape != shape for frame in frames):
        raise ValueError("Foreground frames must share one RGB uint8 raster.")

    model = AutoModelForImageSegmentation.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = model.eval().half().to(device)
    mean = torch.tensor([.485, .456, .406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    output = np.empty((len(frames), *shape[:2]), dtype=np.uint8)
    try:
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                batch = frames[start:start + batch_size]
                pixels = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2).half().div_(255)
                inputs = F.interpolate(pixels, size=FOREGROUND.image_size,
                                       mode='bilinear', antialias=True).sub_(mean).div_(std)
                predictions = model(inputs)[-1].sigmoid()
                masks = F.interpolate(predictions.float(), size=shape[:2], mode='bilinear')
                output[start:start + len(batch)] = masks.mul_(255).round_().clamp_(0, 255).byte().squeeze(1).cpu().numpy()
                del pixels, inputs, predictions, masks
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output
