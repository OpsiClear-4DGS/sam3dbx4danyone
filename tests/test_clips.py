"""Prove trimming/skipping selects exact frames and preserves them through encoding."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fdanyone.space.clips import video_index, plan_clip, prepare_chunks
from fdanyone.video import write_video, read_rgb_video, decode_canonical_clip


class ClipWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'input.mp4'
        def frames():
            for i in range(500):
                frame = np.zeros((16,24,3), dtype=np.uint8)
                frame[:] = (i%256, i//256, 70)
                yield frame
        write_video(frames(), self.path, Fraction(25), lossless_rgb=True)

    def test_index_cache_and_metadata(self):
        info = video_index(self.path)
        self.assertEqual(info['frames'], 500)
        self.assertEqual(info['duration'], 20.)
        with patch('fdanyone.space.clips.av.open', side_effect=AssertionError('Should use cache')):
            self.assertEqual(video_index(self.path), info)

    def test_trim_skip_chunk_and_pipeline_decode_keep_exact_source_frames(self):
        plan = plan_clip(self.path, .4, 19.8, 2)
        self.assertEqual(plan['kept_frames'], 162)
        self.assertEqual(len(plan['chunks']), 2)
        self.assertEqual(plan['chunks'][-1]['overlap_frames'], 80)
        outputs = prepare_chunks(plan, self.root/'clips')
        union = set()
        for chunk, path in zip(plan['chunks'], outputs):
            expected = list(range(chunk['first'], chunk['last']+1, 3))
            frames = read_rgb_video(path, expected_frames=121)
            actual = [int(f[0,0,0])+256*int(f[0,0,1]) for f in frames]
            self.assertEqual(actual, expected)
            union.update(actual)
            # This is the actual first pipeline step, including FPS/timestamp sampling.
            clip = decode_canonical_clip(path, fps='25')
            self.assertEqual([int(f.rgb[0,0,0])+256*int(f.rgb[0,0,1]) for f in clip.frames], expected)
        self.assertEqual(sorted(union), list(range(10,495,3)))

    def test_boundaries_and_short_selection(self):
        exact = plan_clip(self.path, 0, 242/25, 0)
        self.assertEqual(len(exact['chunks']), 2)
        self.assertTrue(all(c['overlap_frames']==0 for c in exact['chunks']))
        for start, end, skip in [(0,1,0),(4,3,0),(-1,10,0),(0,21,0),(0,20,-1),(0,20,1.5),(0,20,True)]:
            with self.subTest(start=start,end=end,skip=skip), self.assertRaises(ValueError):
                plan_clip(self.path,start,end,skip)

    def test_variable_timestamps_trim_by_time_then_skip_by_frame_index(self):
        info=video_index(self.path)
        # Deliberately nonuniform times; don't infer indices by multiplying FPS.
        info['timestamps']=[i*.03 if i<100 else 3+(i-100)*.05 for i in range(500)]
        info['duration']=info['timestamps'][-1]+.05
        with patch('fdanyone.space.clips.video_index',return_value=info):
            plan=plan_clip(self.path,2.4,15,1)
        self.assertEqual(plan['chunks'][0]['first'],80)
        self.assertEqual(plan['chunks'][0]['last'],320)
        self.assertEqual(plan['kept_frames'],130)

    def test_launcher_prepares_chunks_and_preserves_source_fps_for_generation(self):
        from fdanyone.io import write_json
        from fdanyone.space.jobs import main
        request = self.root/'request.json'
        plan = plan_clip(self.path, .4, 19.8, 2)
        write_json(request, dict(video_paths=[str(self.path)], output_dir=str(self.root/'results'),
                                 start_time=1, target_fps='auto', _clip_plan=plan))
        with patch('sys.argv',['jobs',str(request)]), patch('signal.signal'), \
             patch('fdanyone.streams.batch_inference') as batch:
            main()
        args = batch.call_args.kwargs
        self.assertEqual(args['start_time'],0)
        self.assertEqual(args['target_fps'],'25/1')
        self.assertEqual(len(args['video_paths']),2)
        self.assertNotIn('_clip_plan',args)
        self.assertTrue(all(Path(p).is_file() for p in args['video_paths']))

    def test_changed_source_is_rejected(self):
        plan=plan_clip(self.path)
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            prepare_chunks(plan,self.root/'clips')
