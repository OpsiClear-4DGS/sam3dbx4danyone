"""Check that optimized decoding leaves source encoding and CPU attention valid."""
import unittest
import tempfile
from pathlib import Path
from fractions import Fraction
import json
import subprocess
from unittest.mock import Mock, patch
import torch
from fdanyone.config import FULL_EXECUTION
from fdanyone.model.decode_worker import publish_isolated
from fdanyone.model.vae import VaeExecutor
from fdanyone.vendor.diffsynth.models import wan_video_dit as dit


class SpeedDefaultsTests(unittest.TestCase):
    def test_decode_tiles_do_not_change_source_encoding(self):
        model = Mock(upsampling_factor=16)
        executor = VaeExecutor(model, ('cuda:0',), FULL_EXECUTION)
        value = torch.zeros(3, 1, 2, 2)
        executor._encode_view(model, value, 'cuda:0')
        self.assertEqual(model.tiled_encode.call_args.args[2:], ((832, 480), (416, 240)))
        executor._decode_view(model, value, 'cuda:0')
        self.assertEqual(model.tiled_decode.call_args.args[2:], ((52, 44), (28, 44)))

    def test_decode_worker_preserves_device_profile_and_cleans_scratch(self):
        def execute(command, **kwargs):
            work = Path(command[-1])
            request = json.loads((work/'request.json').read_text())
            self.assertEqual(request['devices'], ['cuda:0'])
            self.assertEqual(request['fps'], '25')
            self.assertEqual(request['execution']['vae_decode_tile_size'], [52, 44])
            self.assertTrue(kwargs['check'])
            self.assertTrue(torch.equal(torch.load(work/'latents.pt', weights_only=True), latents))
            (work/'report.json').write_text(json.dumps({'videos': ['view.mp4'], 'peak_vram': {'cuda:0': {'process': 123}}}))
        latents = torch.zeros(1, 48, 1, 2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('fdanyone.model.decode_worker.subprocess.run', side_effect=execute):
                paths, peak = publish_isolated(latents, root/'vae.pth', root/'out', Fraction(25), ('cuda:0',), FULL_EXECUTION)
            self.assertEqual(paths, (Path('view.mp4'),))
            self.assertEqual(peak['cuda:0']['process'], 123)
            self.assertFalse(list(root.glob('.decode-*')))
            with patch('fdanyone.model.decode_worker.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'worker')):
                with self.assertRaises(subprocess.CalledProcessError):
                    publish_isolated(latents, root/'vae.pth', root/'out', Fraction(25), ('cuda:0',), FULL_EXECUTION)
            self.assertFalse(list(root.glob('.decode-*')))

    def test_sage_is_preferred_but_cpu_uses_sdpa(self):
        with patch.object(dit, 'SAGE_ATTN_AVAILABLE', True), patch.object(dit, 'FLASH_ATTN_3_AVAILABLE', True):
            self.assertEqual(dit.get_attention_backend(), 'sageattention')
            q = torch.randn(1, 3, 128)
            with patch.object(dit, 'sageattn', create=True) as sage:
                result = dit.attention(q, q, q, 1)
                sage.assert_not_called()
                self.assertTrue(torch.isfinite(result).all())

    def test_decoder_processes_isolate_gpus_and_preserve_global_camera_order(self):
        def execute(command, **kwargs):
            work = Path(command[-1])
            request = json.loads((work/'request.json').read_text())
            self.assertEqual(request['devices'], ['cuda:0'])
            gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
            self.assertIn(gpu, ('GPU-alpha', 'GPU-beta'))
            latents = torch.load(work/'latents.pt', weights_only=True)
            video_root = Path(request['output_dir']) / 'videos'
            video_root.mkdir(parents=True)
            paths = []
            for local_index, value in enumerate(latents):
                path = video_root / f'{local_index:02d}.mp4'
                path.write_text(str(int(value.item())))
                paths.append(str(path))
            (work/'report.json').write_text(json.dumps(dict(
                videos=paths, peak_vram={'cuda:0': {'process': 100 if gpu == 'GPU-alpha' else 200}})))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-alpha,GPU-beta'}), \
                 patch('fdanyone.model.decode_worker.subprocess.run', side_effect=execute):
                videos, peaks = publish_isolated(
                    torch.arange(5).reshape(5, 1), root/'vae.pth', root/'out',
                    Fraction(25), ('cuda:0', 'cuda:1'), FULL_EXECUTION)
            self.assertEqual([p.name for p in videos], [f'{i:02d}.mp4' for i in range(5)])
            self.assertEqual([p.read_text() for p in videos], list('01234'))
            self.assertEqual(peaks, {'cuda:0': {'process': 100}, 'cuda:1': {'process': 200}})
            self.assertFalse(list(root.glob('.decode-*')))


if __name__ == '__main__':
    unittest.main()
