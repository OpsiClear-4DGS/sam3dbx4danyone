"""Snapshot publication must preserve training and never advertise partial files."""
import json
from pathlib import Path
import tempfile
import unittest

from fdanyone.reconstruction.training_preview import PreviewPublisher, clear_previews


class TrainingPreviewTests(unittest.TestCase):
    def test_cadence_complete_publication_and_bounded_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);status=root/'status.json'
            status.write_text(json.dumps({'status':'running','step':1}))
            now=[0]
            def export(path):
                self.assertNotEqual(json.loads(status.read_text()).get('preview_model'),str(path.relative_to(root)))
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'complete animated model')
            publisher=PreviewPublisher(root,status,export,clock=lambda:now[0])
            self.assertFalse(publisher.publish(51))
            self.assertTrue(publisher.publish(101))
            now[0]=29
            self.assertFalse(publisher.publish(1001))
            now[0]=30
            self.assertTrue(publisher.publish(1501))
            now[0]=60
            self.assertTrue(publisher.publish(3001))
            state=json.loads(status.read_text())
            self.assertEqual(state['preview_step'],3001)
            self.assertEqual((root/state['preview_model']).read_bytes(),b'complete animated model')
            self.assertEqual(len(list((root/'previews').glob('*.ply'))),2)
            (root/'scene.ftgs.ply').write_bytes(b'final model')
            clear_previews(root)
            self.assertFalse((root/'previews').exists())
            self.assertEqual((root/'scene.ftgs.ply').read_bytes(),b'final model')

    def test_failed_export_keeps_previous_advertised_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);status=root/'status.json'
            original={'status':'running','step':501,'preview_step':101,'preview_model':'previous.ftgs.ply'}
            status.write_text(json.dumps(original))
            def export(path):
                raise OSError('disk write interrupted')
            publisher=PreviewPublisher(root,status,export)
            self.assertFalse(publisher.publish(501))
            self.assertEqual(json.loads(status.read_text()),original)
