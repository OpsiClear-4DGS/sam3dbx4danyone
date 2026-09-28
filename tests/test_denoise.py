from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from fdanyone.model.denoise import denoise_group


class _Model:
    def __call__(self, **values):
        return torch.full_like(values["x"], 0.125, dtype=torch.bfloat16)


class _Scheduler:
    timesteps = torch.tensor([500.0])

    def __init__(self) -> None:
        self.prediction_dtype = None

    def step(self, prediction, _timestep, latents):
        self.prediction_dtype = prediction.dtype
        return latents + prediction * 0.25


class DenoiseAccumulationTests(unittest.TestCase):
    def test_prediction_is_promoted_to_fp32_accumulator(self) -> None:
        scheduler = _Scheduler()
        denoiser = SimpleNamespace(
            dtype=torch.bfloat16,
            model=_Model(),
            scheduler=scheduler,
        )
        latents = torch.ones(1, 2, dtype=torch.float32)

        result = denoise_group(
            denoiser,
            latents,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            0,
        )

        self.assertEqual(scheduler.prediction_dtype, torch.float32)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full_like(result, 1.03125))

    def test_bfloat16_release_accumulator_stays_bfloat16(self) -> None:
        scheduler = _Scheduler()
        denoiser = SimpleNamespace(
            dtype=torch.bfloat16,
            model=_Model(),
            scheduler=scheduler,
        )
        latents = torch.ones(1, 2, dtype=torch.bfloat16)

        result = denoise_group(
            denoiser,
            latents,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            0,
        )

        self.assertEqual(scheduler.prediction_dtype, torch.bfloat16)
        self.assertEqual(result.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
