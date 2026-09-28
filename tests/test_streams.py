"""Scheduling checks use real lightweight child processes, without model loads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fdanyone.errors import ConfigurationError
from fdanyone.streams import run_queue, worker_environment, normalize_gpu_uuid, batch_inference


class StreamTests(unittest.TestCase):
    def test_uuid_normalization(self):
        self.assertEqual(normalize_gpu_uuid('abc-def'), 'GPU-abc-def')
        self.assertEqual(normalize_gpu_uuid('GPU-abc-def'), 'GPU-abc-def')
        self.assertEqual(normalize_gpu_uuid('MIG-abc-def'), 'MIG-abc-def')

    def test_invalid_batch_inputs_fail_before_cuda_or_outputs(self):
        for paths in ([], 'video.mp4'):
            with self.assertRaises(ConfigurationError):
                batch_inference(paths)
        with self.assertRaisesRegex(ConfigurationError, 'reserved'):
            batch_inference(['video.mp4'], gpu_ids=[0])
        with self.assertRaisesRegex(ConfigurationError, 'reserved'):
            batch_inference(['video.mp4'], gpu_groups=[[0]])
        with self.assertRaisesRegex(ConfigurationError, 'does not exist'):
            batch_inference(['/nonexistent/stream-test.mp4'])

    def test_batch_automatically_uses_every_visible_gpu(self):
        cuda = SimpleNamespace(device_count=lambda: 8,
                               get_device_properties=lambda i: SimpleNamespace(uuid=f'device-{i}'))
        def complete(jobs, gpus, report_path):
            for job in jobs:
                job['status'] = 'completed'
            return {'gpus': gpus, 'jobs': jobs}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root/'clip.mp4'
            video.touch()
            with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}), \
                 patch('fdanyone.download.ensure_models', return_value=root/'models'), \
                 patch('fdanyone.download.ensure_sam3d'), \
                 patch('fdanyone.download.ensure_turbo'), \
                 patch('fdanyone.streams.run_queue', side_effect=complete) as queue:
                batch_inference([str(video)]*10, output_dir=str(root/'output'))
            jobs, gpus, _ = queue.call_args.args
            self.assertEqual(gpus, [f'GPU-device-{i}' for i in range(8)])
            self.assertEqual(len(jobs), 10)
            self.assertEqual(len({j['result_dir'] for j in jobs}), 10)
            for job in jobs:
                self.assertEqual(json.loads(Path(job['request']).read_text())['gpu_ids'], [0])

    def test_environment_preserves_allocator_and_limits_visibility(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7,4', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}):
            env = worker_environment('GPU-selected')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-selected')
        self.assertEqual(env['PYTORCH_CUDA_ALLOC_CONF'], 'expandable_segments:True')

    def test_concurrent_jobs_reuse_slots_and_continue_after_failure(self):
        real_popen = subprocess.Popen
        script = '''import os,sys,time,json
from pathlib import Path
for line in sys.stdin:
 job=json.loads(line)
 start=time.time()
 time.sleep(.4)
 Path(job['request']+'.observed').write_text(json.dumps([start,time.time(),os.environ['CUDA_VISIBLE_DEVICES']]))
 code=3 if job['request'].endswith('0') else 0
 temporary=Path(job['completion']+'.tmp')
 temporary.write_text(json.dumps({'returncode':code}))
 temporary.replace(job['completion'])
'''
        def launch(command, **kwargs):
            return real_popen([sys.executable, '-c', script, command[-1]], **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [dict(request=str(root/str(i)), log=str(root/f'{i}.log'), status='queued') for i in range(4)]
            with patch('fdanyone.streams.subprocess.Popen', side_effect=launch):
                report = run_queue(jobs, ['GPU-a', 'GPU-b'], root/'report.json')
            observations = [json.loads((root/f'{i}.observed').read_text()) for i in range(4)]
            self.assertEqual(observations[1][2], 'GPU-b')
            self.assertEqual([j['status'] for j in jobs], ['failed', 'completed', 'completed', 'completed'])
            self.assertEqual(jobs[1]['pid'], jobs[3]['pid'])
            self.assertNotEqual(jobs[0]['pid'], jobs[2]['pid'])
            self.assertLess(max(observations[0][0], observations[1][0]), min(observations[0][1], observations[1][1]))
            for slot in range(2):
                first, second = observations[slot], observations[slot+2]
                self.assertEqual(first[2], second[2])
                self.assertGreaterEqual(second[0], first[1])
            self.assertEqual(json.loads((root/'report.json').read_text()), report)

    def test_real_worker_reuses_process_and_separates_job_logs(self):
        real_popen = subprocess.Popen
        script = '''import sys,types
calls=0
def inference(label):
 global calls
 calls+=1
 print(label,flush=True)
 return {'calls':calls,'label':label}
sys.modules['inference']=types.SimpleNamespace(inference=inference)
from fdanyone.streams import _serve_worker
_serve_worker()
'''
        def launch(command, **kwargs):
            return real_popen([sys.executable, '-c', script], **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for index in range(2):
                folder = root / str(index)
                folder.mkdir()
                request = folder / 'request.json'
                request.write_text(json.dumps({'label': f'video-{index}'}))
                jobs.append(dict(request=str(request), log=str(folder/'worker.log'), status='queued'))
            with patch('fdanyone.streams.subprocess.Popen', side_effect=launch):
                run_queue(jobs, ['GPU-a'], root/'report.json')
            self.assertEqual(jobs[0]['pid'], jobs[1]['pid'])
            for index in range(2):
                self.assertEqual((root/str(index)/'worker.log').read_text().strip(), f'video-{index}')
                self.assertEqual(json.loads((root/str(index)/'result.json').read_text())['calls'], index+1)

    def test_interruption_cancels_active_and_pending_jobs(self):
        real_popen = subprocess.Popen
        children = []
        def launch(command, **kwargs):
            child = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            children.append(child)
            return child
        real_sleep = __import__('time').sleep
        def interrupt_poll(seconds):
            if seconds == .2:
                raise KeyboardInterrupt
            real_sleep(seconds)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [dict(request=str(root/str(i)), log=str(root/f'{i}.log'), status='queued') for i in range(2)]
            with patch('fdanyone.streams.subprocess.Popen', side_effect=launch), patch('fdanyone.streams.time.sleep', side_effect=interrupt_poll):
                with self.assertRaises(KeyboardInterrupt):
                    run_queue(jobs, ['GPU-a'], root/'report.json')
            self.assertTrue(all(child.poll() is not None for child in children))
            self.assertEqual([job['status'] for job in jobs], ['cancelled', 'cancelled'])


if __name__ == '__main__':
    unittest.main()
