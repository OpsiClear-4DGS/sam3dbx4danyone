from __future__ import annotations

import unittest

import torch

from fdanyone.vendor.diffsynth.models.wan_video_dit import (
    CrossAttention,
    DiTBlock,
    RMSNorm,
    SelfAttention,
    gated_residual,
    modulate,
    precompute_freqs_cis,
    residual_add,
    rope_apply,
)


class DiTFeedForwardChunkTests(unittest.TestCase):
    @staticmethod
    def _small_block(chunk_views: int | None) -> DiTBlock:
        block = DiTBlock.__new__(DiTBlock)
        torch.nn.Module.__init__(block)
        block.ffn_chunk_views = chunk_views
        block.ffn = torch.nn.Sequential(
            torch.nn.Linear(8, 13),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(13, 8),
        )
        return block

    def test_view_chunks_match_full_feed_forward(self) -> None:
        torch.manual_seed(7)
        full = self._small_block(None).eval()
        chunked = self._small_block(2).eval()
        chunked.load_state_dict(full.state_dict())
        value = torch.randn(7, 11, 8)

        with torch.inference_mode():
            expected = full._feed_forward(value.clone())
            actual_input = value.clone()
            actual = chunked._feed_forward(actual_input)

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_grad_enabled_path_stays_unchunked(self) -> None:
        torch.manual_seed(8)
        full = self._small_block(None)
        chunked = self._small_block(1)
        chunked.load_state_dict(full.state_dict())
        value = torch.randn(3, 5, 8, requires_grad=True)

        expected = full._feed_forward(value)
        actual = chunked._feed_forward(value)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


class DiTInPlaceInferenceTests(unittest.TestCase):
    def test_modulation_matches_out_of_place_path_and_reuses_input(self) -> None:
        torch.manual_seed(9)
        value = torch.randn(2, 7, 8)
        shift = torch.randn(2, 1, 8)
        scale = torch.randn(2, 1, 8)
        expected = modulate(value.clone(), shift, scale)
        actual_input = value.clone()

        with torch.inference_mode():
            actual = modulate(actual_input, shift, scale)

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        self.assertTrue(torch.equal(actual, expected))

    def test_residual_helpers_match_out_of_place_paths_and_reuse_input(self) -> None:
        torch.manual_seed(10)
        value = torch.randn(2, 7, 8)
        residual = torch.randn(2, 7, 8)
        gate = torch.randn(2, 1, 8)
        expected_gated = gated_residual(value.clone(), residual.clone(), gate)
        expected_plain = residual_add(value.clone(), residual.clone())
        gated_input = value.clone()
        plain_input = value.clone()

        with torch.inference_mode():
            actual_gated = gated_residual(gated_input, residual.clone(), gate)
            actual_plain = residual_add(plain_input, residual.clone())

        self.assertEqual(actual_gated.data_ptr(), gated_input.data_ptr())
        self.assertEqual(actual_plain.data_ptr(), plain_input.data_ptr())
        self.assertTrue(torch.equal(actual_gated, expected_gated))
        self.assertTrue(torch.equal(actual_plain, expected_plain))

    def test_rmsnorm_matches_out_of_place_path_and_reuses_input(self) -> None:
        torch.manual_seed(11)
        norm = RMSNorm(8, eps=1e-6).eval()
        norm.weight.data.copy_(torch.randn_like(norm.weight))
        value = torch.randn(2, 13, 8)
        expected = norm(value.clone())
        actual_input = value.clone()

        with torch.inference_mode():
            actual = norm(actual_input)

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        self.assertTrue(torch.equal(actual, expected))

    def test_rope_matches_out_of_place_path_and_reuses_input(self) -> None:
        torch.manual_seed(12)
        value = torch.randn(2, 13, 16)
        frequencies = precompute_freqs_cis(8, end=value.shape[1]).view(
            value.shape[1], 1, -1
        )
        expected = rope_apply(value.clone(), frequencies, num_heads=2)
        actual_input = value.clone()

        with torch.inference_mode():
            actual = rope_apply(actual_input, frequencies, num_heads=2)

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA is required for production DiT parity tests"
)
class DiTCudaBfloat16ParityTests(unittest.TestCase):
    def test_ffn_view_chunks_are_exact(self) -> None:
        torch.manual_seed(16)
        full = (
            DiTFeedForwardChunkTests._small_block(None)
            .to("cuda", torch.bfloat16)
            .eval()
        )
        chunked = (
            DiTFeedForwardChunkTests._small_block(2).to("cuda", torch.bfloat16).eval()
        )
        chunked.load_state_dict(full.state_dict())
        value = torch.randn(7, 11, 8, device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            expected = full._feed_forward(value.clone())
            actual = chunked._feed_forward(value.clone())

        self.assertTrue(torch.equal(actual, expected))

    def test_cross_attention_batches_are_exact(self) -> None:
        torch.manual_seed(17)
        full = CrossAttention(128, 4).to("cuda", torch.bfloat16).eval()
        chunked = (
            CrossAttention(128, 4, batch_limit=1).to("cuda", torch.bfloat16).eval()
        )
        chunked.load_state_dict(full.state_dict())
        value = torch.randn(4, 257, 128, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            expected = full(value.clone(), context.clone())
            actual = chunked(value.clone(), context.clone())

        self.assertTrue(torch.equal(actual, expected))


class DiTAttentionBatchChunkTests(unittest.TestCase):
    def test_self_attention_batches_match_full_inference(self) -> None:
        torch.manual_seed(13)
        full = SelfAttention(8, 2).eval()
        chunked = SelfAttention(8, 2, batch_limit=1).eval()
        chunked.load_state_dict(full.state_dict())
        value = torch.randn(4, 11, 8)
        frequencies = precompute_freqs_cis(4, end=value.shape[1]).view(
            value.shape[1], 1, -1
        )
        expected = full(value.clone(), frequencies)
        actual_input = value.clone()

        with torch.inference_mode():
            actual = chunked(actual_input, frequencies)

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        self.assertTrue(torch.equal(actual, expected))

    def test_cross_attention_batches_match_full_inference(self) -> None:
        torch.manual_seed(14)
        full = CrossAttention(8, 2).eval()
        chunked = CrossAttention(8, 2, batch_limit=1).eval()
        chunked.load_state_dict(full.state_dict())
        value = torch.randn(4, 11, 8)
        context = torch.randn(4, 5, 8)
        expected = full(value.clone(), context.clone())
        actual_input = value.clone()

        with torch.inference_mode():
            actual = chunked(actual_input, context.clone())

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_cross_attention_projects_shared_context_once(self) -> None:
        torch.manual_seed(15)
        attention = CrossAttention(8, 2, batch_limit=1).eval()
        value = torch.randn(4, 11, 8)
        context = torch.randn(1, 5, 8)

        with torch.inference_mode():
            expected = attention(value.clone(), context.expand(4, -1, -1).clone())
            actual_input = value.clone()
            actual = attention(actual_input, context.clone())

        self.assertEqual(actual.data_ptr(), actual_input.data_ptr())
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
