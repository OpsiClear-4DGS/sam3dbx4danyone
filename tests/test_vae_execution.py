from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fdanyone.config import FULL_EXECUTION
from fdanyone.errors import FourDAnyoneError
from fdanyone.model.inference import _target_decode_devices
from fdanyone.model.loader import load_vae
from fdanyone.model.vae import VaeExecutor, _rgb_frames


class _FakeVae:
    upsampling_factor = 8

    def __init__(self) -> None:
        self.call = None

    def tiled_encode(self, value, device, tile_size, tile_stride):
        self.call = ("tiled_encode", value, device, tile_size, tile_stride)
        return value

    def tiled_decode(self, value, device, tile_size, tile_stride):
        self.call = ("tiled_decode", value, device, tile_size, tile_stride)
        return value


class VaeExecutionProfileTests(unittest.TestCase):
    def test_target_decode_avoids_primary_dit_device_when_possible(self) -> None:
        self.assertEqual(
            _target_decode_devices(("cuda:0", "cuda:1", "cuda:2"), "cuda:0"),
            ("cuda:1", "cuda:2"),
        )
        self.assertEqual(_target_decode_devices(("cuda:0",), "cuda:0"), ("cuda:0",))

    def test_release_replicas_refreshes_the_cross_stage_prototype(self) -> None:
        original = _FakeVae()
        executor = VaeExecutor(original, ("cpu",), FULL_EXECUTION)
        executor._models.append(_FakeVae())

        executor.release_replicas()

        self.assertEqual(len(executor._models), 1)
        self.assertIsNot(executor._models[0], original)

    def test_full_profile_uses_scaled_encode_tiles(self) -> None:
        model = _FakeVae()
        executor = VaeExecutor(model, ("cpu",), FULL_EXECUTION)
        executor._encode_view(model, torch.zeros(3, 2, 4, 4), "cpu")

        self.assertEqual(model.call[0], "tiled_encode")
        self.assertEqual(model.call[1].dtype, torch.bfloat16)
        self.assertEqual(model.call[3], (416, 240))
        self.assertEqual(model.call[4], (208, 120))

    def test_full_profile_uses_latent_decode_tiles(self) -> None:
        model = _FakeVae()
        executor = VaeExecutor(model, ("cpu",), FULL_EXECUTION)
        executor._decode_view(model, torch.zeros(48, 2, 4, 4), "cpu")

        self.assertEqual(model.call[0], "tiled_decode")
        self.assertEqual(model.call[1].dtype, torch.float16)
        self.assertEqual(model.call[3], FULL_EXECUTION.vae_decode_tile_size)
        self.assertEqual(model.call[4], FULL_EXECUTION.vae_decode_tile_stride)

    def test_rgb_conversion_rounds_clamps_and_preserves_input(self) -> None:
        values = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        video = values.reshape(1, 1, 1, -1).repeat(3, 2, 1, 1)
        original = video.clone()
        frames = _rgb_frames(video)
        self.assertTrue(torch.equal(video, original))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].dtype, np.uint8)
        self.assertEqual(frames[0][0, :, 0].tolist(), [0, 0, 128, 191, 255, 255])

    def test_nonfinite_pixels_fail_before_video_export(self) -> None:
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(FourDAnyoneError, 'non-finite'):
                _rgb_frames(torch.full((3, 1, 2, 2), value))

    def test_decoder_weights_are_not_rounded_through_bf16(self) -> None:
        class TinyVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                for name in ('encoder', 'decoder', 'conv1', 'conv2'):
                    setattr(self.model, name, torch.nn.Linear(1, 1, bias=False))

            @staticmethod
            def state_dict_converter():
                return SimpleNamespace(from_civitai=lambda state: state)

            def materialize_normalization(self, device):
                pass

        weights = {f'model.{name}.weight': torch.full((1, 1), 1.003)
                   for name in ('encoder', 'decoder', 'conv1', 'conv2')}
        with patch('torch.load', return_value=weights), patch(
            'fdanyone.vendor.diffsynth.models.wan_video_vae.WanVideoVAE38', TinyVae
        ):
            vae = load_vae(Path('vae.pth'))
        for name in ('encoder', 'conv1'):
            self.assertEqual(getattr(vae.model, name).weight.dtype, torch.bfloat16)
        for name in ('decoder', 'conv2'):
            actual = getattr(vae.model, name).weight
            original = weights[f'model.{name}.weight']
            self.assertEqual(actual.dtype, torch.float16)
            self.assertTrue(torch.equal(actual, original.half()))
            self.assertFalse(torch.equal(actual, original.bfloat16().half()))
            self.assertFalse(actual.requires_grad)



if __name__ == "__main__":
    unittest.main()
