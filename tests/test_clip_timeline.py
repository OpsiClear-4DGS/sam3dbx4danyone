"""Check browser timeline previews and snap targets against the actual clip planner."""
import json
from pathlib import Path
import random
import shutil
import subprocess
import unittest
from unittest.mock import patch

from fdanyone.space.clips import plan_clip


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser timeline checks.')
class ClipTimelineTests(unittest.TestCase):
    def browser(self, requests):
        module = (Path(__file__).resolve().parents[1]/'fdanyone/space/static/clip-timeline.js').as_uri()
        script = """
            import {readFileSync} from 'node:fs';
            const {previewClip, snapClipEnd} = await import(process.argv[1]);
            const requests = JSON.parse(readFileSync(0, 'utf8'));
            process.stdout.write(JSON.stringify(requests.map(({info,edit,width=1000})=>({
                preview:previewClip(info,edit), snap:snapClipEnd(info,edit,width)
            }))));
        """
        result = subprocess.run(['node', '--input-type=module', '-e', script, module],
                                input=json.dumps(requests), text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def info(self, times, fps_num=25, fps_den=1):
        return dict(timestamps=times, frames=len(times), duration=times[-1]+fps_den/fps_num,
                    fps_num=fps_num, fps_den=fps_den, chunk_frames=121, identity=[])

    def server(self, info, edit):
        with patch('fdanyone.space.clips.video_index', return_value=info):
            return plan_clip('unused.mp4', **edit)

    def test_previews_match_server_across_speeds_and_frame_rates(self):
        rng = random.Random(17)
        cases = []
        for info in [self.info([i/25 for i in range(1500)]),
                     self.info([i*1001/30000 for i in range(1500)], 30000, 1001),
                     self.info([i*.02 if i<400 else 8+(i-400)*.07 for i in range(1500)])]:
            for skip in range(4):
                for _ in range(15):
                    first = rng.randrange(100)
                    stop = rng.randrange(first+121*(skip+1), info['frames'])
                    edit = dict(start=info['timestamps'][first]+.001,
                                end=info['timestamps'][stop]-.001, skip=skip)
                    cases.append(dict(info=info, edit=edit))
        for case, result in zip(cases, self.browser(cases)):
            plan = self.server(case['info'], case['edit'])
            for key, value in result['preview'].items():
                self.assertEqual(value, plan[key], (case['edit'], key))

    def test_snaps_make_exact_chunks_with_real_timestamps(self):
        cases = []
        for info in [self.info([i/25 for i in range(1600)]),
                     self.info([i*1001/30000 for i in range(1600)], 30000, 1001),
                     self.info([i*.02 if i<400 else 8+(i-400)*.07 for i in range(1600)])]:
            for skip in range(4):
                for first in [0, 19, 403]:
                    for count in [1, 2]:
                        end = info['timestamps'][first+count*121*(skip+1)]
                        for delta in [-.001, .001]:
                            cases.append(dict(info=info, edit=dict(start=info['timestamps'][first],
                                             end=end+delta, skip=skip), expected=end, count=count))
        for case, result in zip(cases, self.browser(cases)):
            self.assertEqual(result['snap'], case['expected'])
            plan = self.server(case['info'], {**case['edit'], 'end': result['snap']})
            self.assertEqual(plan['kept_frames'], case['count']*121)
            self.assertTrue(all(c['overlap_frames']==0 for c in plan['chunks']))

    def test_snap_radius_short_sources_and_end_of_source(self):
        info = self.info([i/25 for i in range(500)])
        cases = [
            dict(info=info, edit=dict(start=0, end=9.75, skip=0), width=1000),
            dict(info=info, edit=dict(start=0, end=10, skip=0), width=1000),
            dict(info=info, edit=dict(start=0, end=10.5, skip=0), width=30),
            dict(info=self.info([i/25 for i in range(120)]), edit=dict(start=0, end=4.79, skip=0)),
            dict(info=self.info([i/25 for i in range(241)]), edit=dict(start=0, end=9.63, skip=1)),
            dict(info=self.info([i/25 for i in range(122)]), edit=dict(start=0, end=4.88, skip=0)),
        ]
        results = self.browser(cases)
        self.assertEqual([r['snap'] for r in results], [9.68, None, None, None, cases[4]['info']['duration'], None])
        self.assertIsNone(results[3]['preview'])
        plan = self.server(cases[4]['info'], {**cases[4]['edit'], 'end': results[4]['snap']})
        self.assertEqual(plan['kept_frames'], 121)

    def test_duplicate_timestamps_do_not_claim_an_unreachable_snap(self):
        times = [i/25 for i in range(500)]
        times[121] = times[120]
        result = self.browser([dict(info=self.info(times), edit=dict(start=0, end=4.82, skip=0))])[0]
        self.assertIsNone(result['snap'])
