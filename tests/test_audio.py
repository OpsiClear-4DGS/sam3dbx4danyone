"""Audio must follow actual trim/skip boundaries, not the unedited upload."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from fdanyone.audio import extract_audio, export_clip_audio
from fdanyone.space.clips import plan_clip, prepare_chunks
from fdanyone.video import write_video, decode_canonical_clip


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for audio export.')
class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.silent = self.root/'silent.mp4'
        write_video((np.zeros((16,24,3),np.uint8) for _ in range(500)), self.silent, Fraction(25), lossless_rgb=True)
        self.source = self.root/'source.mp4'
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(self.silent),'-f','lavfi','-i',
                        'aevalsrc=0.3*sin(2*PI*(200*t+10*t*t)):s=48000:d=20',
                        '-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','192k',str(self.source)], check=True)

    def read(self, path):
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            duration = float(stream.duration*stream.time_base)
            samples = np.concatenate([frame.to_ndarray()[0] for frame in container.decode(stream)])
        return samples, duration

    def frequency(self, samples, time):
        window = samples[int((time-.15)*48000):int((time+.15)*48000)]
        spectrum = np.abs(np.fft.rfft(window*np.hanning(len(window))))
        return np.fft.rfftfreq(len(window),1/48000)[spectrum.argmax()]

    def test_trim_and_four_times_tempo_preserve_pitch_and_duration(self):
        output = self.root/'audio.m4a'
        extract_audio(self.source, output, start=2, end=18, duration=4)
        samples, duration = self.read(output)
        self.assertAlmostEqual(duration,4,delta=.025)
        for t in (.5,2,3.5):
            self.assertAlmostEqual(self.frequency(samples,t),200+20*(2+4*t),delta=12)

    def test_chunk_overlap_audio_follows_each_source_start_and_is_reused(self):
        plan = plan_clip(self.source,.4,19.8,2)
        paths = prepare_chunks(plan,self.root/'chunks')
        self.assertEqual(len(paths),2)
        for chunk,path in zip(plan['chunks'],paths,strict=True):
            audio = Path(path).with_suffix('.audio.m4a')
            samples,duration = self.read(audio)
            self.assertAlmostEqual(duration,4.8,delta=.025)
            self.assertAlmostEqual(self.frequency(samples,1),200+20*(chunk['start']+3),delta=12)
            clip = decode_canonical_clip(path,fps='25')
            published = self.root/(Path(path).stem+'.m4a')
            export_clip_audio(clip,published)
            self.assertEqual(published.read_bytes(),audio.read_bytes())

    def test_silent_source_and_failed_encoder_do_not_publish_audio(self):
        output = self.root/'audio.m4a'
        self.assertIsNone(extract_audio(self.silent,output,start=0,end=4,duration=4))
        with patch('fdanyone.audio.subprocess.run',side_effect=subprocess.CalledProcessError(1,'ffmpeg')):
            with self.assertRaises(subprocess.CalledProcessError):
                extract_audio(self.source,output,start=0,end=4,duration=4)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.audio-*')))
