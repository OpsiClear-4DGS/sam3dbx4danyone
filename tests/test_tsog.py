"""Final container publication is atomic and cannot replace a previous model."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fdanyone.reconstruction.tsog import export_tsog


class TSOGPublicationTests(unittest.TestCase):
    def test_failed_encoder_preserves_source_and_leaves_no_partial_container(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'scene.ftgs.ply';source.write_bytes(b'original')
            target=Path(temp)/'scene.tsog'
            with patch('fdanyone.reconstruction.tsog.encoder_runtime',return_value='node'), \
                 patch('fdanyone.reconstruction.tsog.subprocess.run',side_effect=subprocess.CalledProcessError(1,'encoder')):
                with self.assertRaises(subprocess.CalledProcessError):
                    export_tsog(source,target,fps='30000/1001')
            self.assertEqual(source.read_bytes(),b'original')
            self.assertFalse(target.exists())
            self.assertFalse(list(Path(temp).glob('.tsog-*')))

    def test_complete_package_published_once_and_audio_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'scene.ftgs.ply';source.write_bytes(b'original')
            target=root/'scene.tsog';audio=root/'audio.m4a';audio.write_bytes(b'audio')
            def encode(args,**_):
                self.assertFalse(target.exists())
                self.assertEqual(args[-1],str(audio))
                with zipfile.ZipFile(args[3],'w') as package:
                    package.writestr('meta.json',json.dumps(dict(version=4,count=32,timeline=dict(type=1))))
            with patch('fdanyone.reconstruction.tsog.encoder_runtime',return_value='node'), \
                 patch('fdanyone.reconstruction.tsog.subprocess.run',side_effect=encode):
                report=export_tsog(source,target,fps='25/1',audio=audio)
                self.assertEqual(report['count'],32)
                with self.assertRaises(FileExistsError):export_tsog(source,target,fps=25)
            self.assertEqual(source.read_bytes(),b'original')
