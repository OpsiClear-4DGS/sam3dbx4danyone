from __future__ import annotations

import unittest
from inspect import signature

from fdanyone.config import (
    FULL_EXECUTION,
    INFERENCE,
    resolve_execution_profile,
)
from fdanyone.errors import ConfigurationError
from inference import inference


class ExecutionProfileTests(unittest.TestCase):
    def test_full_limits(self) -> None:
        profile = resolve_execution_profile(" FULL ")
        self.assertIs(profile, FULL_EXECUTION)
        self.assertEqual(profile.pose_encoder_batch_limit, 1)
        self.assertTrue(profile.pose_encoder_separate_nulls)
        self.assertEqual(profile.dit_ffn_chunk_views, 1)
        self.assertEqual(profile.dit_attention_batch_limit, 1)
        self.assertEqual(profile.pose_encoder_worker_limit, 1)
        self.assertTrue(profile.cuda_expandable_segments)
        self.assertEqual(profile.cuda_garbage_collection_threshold, 0.70)
        self.assertTrue(profile.tiled_vae)
        self.assertEqual(profile.process_vram_limit_bytes, 23 * 2**30)

    def test_retired_profiles_are_rejected(self) -> None:
        for name in ("balanced", "low-vram", "LOW_VRAM"):
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                resolve_execution_profile(name)

    def test_quality_profile_spends_headroom_on_precision_and_publication(self) -> None:
        profile = resolve_execution_profile("quality")
        self.assertIs(profile, FULL_EXECUTION)
        self.assertEqual(profile.to_dict()["name"], "full")
        self.assertEqual(profile.latent_accumulation_dtype, "float32")
        self.assertTrue(profile.lossless_target_video)
        self.assertEqual(profile.target_video_preset, "veryfast")
        self.assertTrue(profile.tiled_vae)
        self.assertEqual(profile.vae_tile_size, INFERENCE.vae_tile_size)
        self.assertEqual(profile.vae_decode_tile_size, (52, 44))
        self.assertEqual(profile.vae_decode_tile_stride, (28, 44))
        self.assertEqual(profile.process_vram_limit_bytes, 23 * 2**30)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "execution_profile"):
            resolve_execution_profile("tiny")

    def test_cli_defaults_to_full_mode(self) -> None:
        parameters = signature(inference).parameters
        self.assertEqual(parameters["execution_profile"].default, "full")
        self.assertEqual(parameters["motion_precision"].default, "fp16")
        self.assertTrue(parameters["turbo"].default)
        self.assertEqual(parameters["views_per_layer"].default, 6)


if __name__ == "__main__":
    unittest.main()
