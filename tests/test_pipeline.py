"""Boundary checks for the active preprocessing pipeline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fdanyone.assets import SAM3D_FILES, resolve_sam3d_assets
from fdanyone.download import ensure_models
from fdanyone.motion.sam3d import mask_boxes
from fdanyone.pipeline import _worker_environment


class ActivePreprocessingTests(unittest.TestCase):
    def test_sam_assets_do_not_require_retired_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in SAM3D_FILES:
                path = root / 'sam3d-body' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            assets = resolve_sam3d_assets(root)
            self.assertTrue(assets.sam3d_checkpoint.is_file())

    def test_mask_box_uses_largest_component_and_rejects_empty(self):
        masks = np.zeros((1, 20, 20), dtype=np.uint8)
        masks[0, 3:15, 5:11] = 255
        masks[0, 0, 0] = 255
        np.testing.assert_array_equal(mask_boxes(masks), [[5, 3, 11, 15]])
        with self.assertRaisesRegex(ValueError, 'Empty BiRefNet'):
            mask_boxes(np.zeros_like(masks))

    def test_worker_environment_is_isolated(self):
        from fdanyone.device import CUDA_ALLOCATOR_CONF
        with patch.dict('os.environ', {CUDA_ALLOCATOR_CONF: 'expandable_segments:True'}):
            environment = _worker_environment()
        self.assertNotIn(CUDA_ALLOCATOR_CONF, environment)
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'], '0')

    def test_model_download_has_no_detector_hook(self):
        with tempfile.TemporaryDirectory() as temporary, patch('fdanyone.download._snapshot') as snapshot, patch('fdanyone.download.ensure_foreground_model'):
            ensure_models(temporary)
            files = snapshot.call_args.args[0]
        self.assertFalse(any('gvhmr' in name or 'vitpose' in name or 'detector' in name for name in files))

if __name__ == '__main__':
    unittest.main()
