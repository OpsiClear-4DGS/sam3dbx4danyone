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

    with ForegroundSegmenter(model_path, device, batch_size=batch_size) as segmenter:
        return segmenter(frames)


class ForegroundSegmenter:
    """Reuse one BiRefNet instance across bounded batches of generated views."""

    def __init__(self, model_path, device, *, batch_size=FOREGROUND.batch_size):
        import torch
        from transformers import AutoModelForImageSegmentation

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(Path(model_path).expanduser().resolve()),
            local_files_only=True, trust_remote_code=True,
        ).eval().half().to(device)
        self.mean = torch.tensor([.485, .456, .406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        import torch

        del self.model, self.mean, self.std
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, frames):
        import torch
        import torch.nn.functional as F

        if not frames:
            raise ValueError("Foreground inference requires at least one frame.")
        shape = frames[0].shape
        if len(shape) != 3 or shape[2] != 3 or any(frame.dtype != np.uint8 or frame.shape != shape for frame in frames):
            raise ValueError("Foreground frames must share one RGB uint8 raster.")
        output = np.empty((len(frames), *shape[:2]), dtype=np.uint8)
        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start:start + self.batch_size]
                pixels = torch.from_numpy(np.stack(batch)).to(self.device).permute(0, 3, 1, 2).half().div_(255)
                inputs = F.interpolate(pixels, size=FOREGROUND.image_size,
                                       mode='bilinear', antialias=True).sub_(self.mean).div_(self.std)
                predictions = self.model(inputs)[-1].sigmoid()
                masks = F.interpolate(predictions.float(), size=shape[:2], mode='bilinear')
                output[start:start + len(batch)] = masks.mul_(255).round_().clamp_(0, 255).byte().squeeze(1).cpu().numpy()
        return output
