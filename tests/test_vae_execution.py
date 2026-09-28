from __future__ import annotations

import unittest

import torch

from fdanyone.config import FULL_EXECUTION
from fdanyone.model.inference import _target_decode_devices
from fdanyone.model.vae import VaeExecutor


class _FakeVae:
    upsampling_factor = 8

    def __init__(self) -> None:
        self.call = None

    def encode_view(self, value):
        self.call = ("encode", value)
        return value

    def tiled_encode(self, value, device, tile_size, tile_stride):
        self.call = ("tiled_encode", value, device, tile_size, tile_stride)
        return value

    def decode_view(self, value):
        self.call = ("decode", value)
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
        self.assertEqual(model.call[3], (416, 240))
        self.assertEqual(model.call[4], (208, 120))

    def test_full_profile_uses_latent_decode_tiles(self) -> None:
        model = _FakeVae()
        executor = VaeExecutor(model, ("cpu",), FULL_EXECUTION)
        executor._decode_view(model, torch.zeros(48, 2, 4, 4), "cpu")

        self.assertEqual(model.call[0], "tiled_decode")
        self.assertEqual(model.call[3], FULL_EXECUTION.vae_decode_tile_size)
        self.assertEqual(model.call[4], FULL_EXECUTION.vae_decode_tile_stride)



if __name__ == "__main__":
    unittest.main()
