"""Memory-policy boundaries and compiled module/state-dict compatibility."""
import unittest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch

from fdanyone.config import GIB, FULL_EXECUTION, resolve_compilation_policy
from fdanyone.model.loader import compile_denoiser, load_denoiser, clear_denoiser_cache


class CompilationTests(unittest.TestCase):
    def policy(self, total=48, free=44, enabled=True, turbo=True):
        return resolve_compilation_policy(
            FULL_EXECUTION, enabled=enabled, turbo=turbo,
            total_memory_bytes=total * GIB, free_memory_bytes=free * GIB)

    def test_large_gpu_uses_compilation_without_changing_precision_or_vae(self):
        result = self.policy()
        self.assertTrue(result.compile_dit)
        self.assertEqual(result.process_vram_limit_bytes, 23 * GIB)
        for name in ('latent_accumulation_dtype', 'vae_tile_size', 'vae_decode_tile_size',
                     'lossless_target_video', 'dit_ffn_chunk_views'):
            self.assertEqual(getattr(result, name), getattr(FULL_EXECUTION, name))
        self.assertEqual(FULL_EXECUTION.process_vram_limit_bytes, 23 * GIB)

    def test_small_busy_disabled_and_base_keep_original_memory_contract(self):
        for kwargs in ({'total': 22}, {'free': 22},
                       {'enabled': False}, {'turbo': False}):
            with self.subTest(**kwargs):
                result = self.policy(**kwargs)
                self.assertFalse(result.compile_dit)
                self.assertEqual(result.process_vram_limit_bytes, 23 * GIB)

    def test_compilation_preserves_weights_names_and_numerical_behavior(self):
        class SmallAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4)
            def _forward_batch(self, value):
                return torch.nn.functional.silu(self.projection(value))
            def forward(self, value):
                return self._forward_batch(value)
        class SmallBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SmallAttention()
                self.self_attn_mvs = SmallAttention()
                self.cross_attn = SmallAttention()
            def forward(self, value):
                return self.cross_attn(self.self_attn_mvs(self.self_attn(value)))
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([SmallBlock()])
            def forward(self, value):
                for block in self.blocks:
                    value = block(value)
                return value
        model = SmallModel().eval().requires_grad_(False)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(2, 4)
        with torch.inference_mode():
            reference = model(x)
            compile_denoiser(model)
            compile_denoiser(model)
            torch.testing.assert_close(model(x), reference)
        self.assertEqual(list(model.state_dict()), list(weights))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_worker_cache_reuses_model_but_invalidates_weight_and_mode_changes(self):
        clear_denoiser_cache()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'model.safetensors'
                path.write_bytes(b'first')
                with patch.dict('os.environ', {'FDANYONE_REUSE_DENOISER': '1'}), \
                     patch('fdanyone.model.loader._load_dit', side_effect=lambda *a, **k: torch.nn.Linear(2, 2)):
                    first = load_denoiser(checkpoint_path=path, cache_identity=('turbo', True))
                    self.assertIs(first, load_denoiser(checkpoint_path=path, cache_identity=('turbo', True)))
                    base = load_denoiser(checkpoint_path=path, cache_identity=('base', False))
                    self.assertIsNot(base, first)
                    path.write_bytes(b'new checkpoint')
                    self.assertIsNot(base, load_denoiser(checkpoint_path=path, cache_identity=('base', False)))
                    clear_denoiser_cache()
                    self.assertIsNot(first, load_denoiser(checkpoint_path=path, cache_identity=('turbo', True)))
        finally:
            clear_denoiser_cache()


if __name__ == '__main__':
    unittest.main()
