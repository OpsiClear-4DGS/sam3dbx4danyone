"""Job lifecycle, lineage, restart recovery and immutable training versions."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from fdanyone.errors import FourDAnyoneError
from fdanyone.io import write_json
from fdanyone.space.jobs import JobManager
from fdanyone.space.server import create_app
from fdanyone.space.workflow import STAGES, config_for, read, run_job, selected_run, stale
from fdanyone.video import write_video


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        assets = self.root/'assets';assets.mkdir()
        self.app = create_app(self.root/'ui', web_assets=assets)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.manager = self.app.state.manager
        self.store = self.manager.workflow
        self.source = self.root/'clip.mp4'
        write_video((np.full((16,24,3),i%256,np.uint8) for i in range(242)), self.source, Fraction(25), lossless_rgb=True)
        self.upload = self.client.post('/api/uploads', files={'file':('Dance.mp4', self.source.read_bytes())}).json()
        self.job_id = self.upload['job_id']
        self.directory = self.manager.root/self.job_id

    def queue(self, start, through, configs=None):
        with patch('fdanyone.space.workflow.subprocess.Popen') as launch, patch('fdanyone.reconstruction.check_runtime'):
            launch.return_value.poll.return_value = None
            self.store.start(self.job_id, start, through, configs or {})
        return self.directory/'job.json'

    def finish(self, start, through, configs=None, fail=None):
        path = self.queue(start, through, configs)
        def execute(job, attempt, model_dir):
            if fail == attempt['stage']:
                raise ValueError('Injected step failure')
            target=self.directory/'stages'/attempt['id'];target.mkdir(parents=True)
            video=target/'chunk.mp4';video.write_bytes(b'clip')
            pose=target/'pose';pose.mkdir();(pose/'sam3d_predictions.npz').write_bytes(b'pose')
            result=target/'generated';result.mkdir();write_json(result/'metadata.json', {})
            training=target/'training';training.mkdir()
            write_json(training/'status.json', {'status':'completed','model':'scene.tsog','steps':100,'step':100})
            attempt['artifacts']=[dict(chunk=0,label='Chunk 1',video=str(video),fps='25/1',
                                       pose_dir=str(pose),result_dir=str(result),training_dir=str(training))]
        with patch('fdanyone.space.workflow.run_stage', side_effect=execute):
            run_job(path, 'models')
        self.manager.process = None
        return self.store.describe(self.job_id)

    def test_upload_creates_draft_job_and_survives_manager_restart(self):
        detail=self.client.get('/api/jobs/'+self.job_id).json()
        self.assertEqual(detail['runs'], [])
        self.assertEqual(detail['source']['name'], 'Dance.mp4')
        self.assertEqual(self.client.get('/api/jobs').json()[0]['state'], 'draft')
        self.assertIsNone(detail['blocked']['trim'])
        self.assertIn('Trim',detail['blocked']['pose'])
        restored=JobManager(self.manager.root).workflow.describe(self.job_id)
        self.assertEqual(restored['id'],detail['id'])
        self.assertEqual(restored['source'],detail['source'])

    def test_start_in_middle_requires_available_inputs(self):
        for stage in ('pose','generate','splat'):
            response=self.client.post(f'/api/jobs/{self.job_id}/run',json={'start':stage,'through':stage})
            self.assertEqual(response.status_code,400,response.text)
        self.assertEqual(read(self.directory/'job.json')['runs'],[])

    def test_versions_keep_lineage_and_selecting_history_restores_ancestors(self):
        first=self.finish('trim','splat')
        first_ids=dict(first['selected'])
        original=deepcopy(first['runs'])
        second=self.finish('generate','generate',{'generate':{'views':8,'seed':99}})
        self.assertEqual(second['selected']['pose'],first_ids['pose'])
        self.assertTrue(selected_run(second,'splat')['stale'])
        self.assertEqual([{k:v for k,v in r.items() if k!='stale'} for r in second['runs'][:4]],
                         [{k:v for k,v in r.items() if k!='stale'} for r in original])
        chosen=self.store.select(self.job_id,first_ids['splat'])
        self.assertEqual(chosen['selected'],first_ids)
        self.assertFalse(any(r['stale'] for r in chosen['runs'] if r['id'] in first_ids.values()))
        self.assertEqual(chosen['configs']['generate']['views'],6)
        latest=second['selected']['generate']
        self.store.select(self.job_id,latest)
        third=self.finish('splat','splat',{'splat':{'steps':100}})
        self.assertEqual(selected_run(third,'splat')['inputs']['generate'],latest)
        self.assertEqual(len(third['runs']),6)
        self.assertNotEqual(selected_run(first,'splat')['artifacts'][0]['training_dir'],selected_run(third,'splat')['artifacts'][0]['training_dir'])

    def test_upstream_rerun_blocks_stale_middle_start(self):
        first=self.finish('trim','generate')
        self.finish('trim','trim',{'trim':{'skip':1}})
        detail=self.store.describe(self.job_id)
        self.assertIn('earlier inputs',detail['blocked']['generate'])
        with self.assertRaisesRegex(ValueError,'earlier inputs'):
            self.queue('generate','generate')
        restored=self.store.select(self.job_id,first['selected']['generate'])
        self.assertIsNone(restored['blocked']['generate'])

    def test_failure_and_cancelled_following_steps_preserve_good_versions(self):
        first=self.finish('trim','generate')
        second=self.finish('pose','splat',fail='pose')
        self.assertEqual(second['selected'],first['selected'])
        self.assertEqual([r['status'] for r in second['runs'][-3:]],['failed','cancelled','cancelled'])
        self.assertIsNone(second['active'])
        self.assertIsNone(second['blocked']['generate'])

    def test_abandoned_process_is_marked_interrupted_and_is_rerunnable(self):
        self.queue('trim','pose')
        self.manager.process=None
        job=JobManager(self.manager.root).workflow.describe(self.job_id)
        self.assertIsNone(job['active'])
        self.assertEqual([r['status'] for r in job['runs']],['interrupted','interrupted'])
        self.assertIsNone(job['blocked']['trim'])

    def test_another_manager_cannot_launch_during_task_file_lock(self):
        import fcntl
        other=JobManager(self.manager.root)
        with (self.manager.root/'.pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'already running'):
                other.workflow.start(self.job_id,'trim','trim',{})
            with self.assertRaisesRegex(ValueError,'active task'):
                other.workflow.configure(self.job_id,{'pose':{'precision':'fp32'}})
        self.assertEqual(read(self.directory/'job.json')['runs'],[])

    def test_active_job_rejects_reconfigure_select_and_delete(self):
        first=self.finish('trim','generate')
        self.queue('generate','generate')
        for method, suffix, body in [('patch','',{'configs':{'generate':{'seed':100}}}),
                                     ('post','/select',{'run_id':first['selected']['generate']}),
                                     ('delete','',None)]:
            kwargs={'json':body} if body else {}
            response=getattr(self.client,method)(f'/api/jobs/{self.job_id}'+suffix,**kwargs)
            self.assertEqual(response.status_code,400,response.text)
        self.manager.process=None

    def test_trim_runs_lossless_and_rerun_does_not_overwrite(self):
        first_path=self.queue('trim','trim')
        run_job(first_path,'models');self.manager.process=None
        first=self.store.describe(self.job_id)
        clip=Path(selected_run(first,'trim')['artifacts'][0]['video'])
        self.assertTrue(clip.is_file())
        self.assertEqual(len(selected_run(first,'trim')['artifacts']),2)
        original=clip.read_bytes()
        second_path=self.queue('trim','trim',{'trim':{'skip':1}})
        run_job(second_path,'models');self.manager.process=None
        second=self.store.describe(self.job_id)
        self.assertEqual(len(selected_run(second,'trim')['artifacts']),1)
        self.assertEqual(clip.read_bytes(),original)
        self.assertNotEqual(clip,Path(selected_run(second,'trim')['artifacts'][0]['video']))

    def test_validation_rejects_bad_configs_and_reversed_ranges(self):
        for stage, values in [('trim',{'skip':4}),('pose',{'precision':'int8'}),('generate',{'views':7}),
                              ('splat',{'steps':0}),('trim',{'start':float('inf')}),('splat',{'output_dir':'/tmp'})]:
            with self.assertRaises((ValueError,TypeError,FourDAnyoneError)):
                config_for(stage,values)
        response=self.client.post(f'/api/jobs/{self.job_id}/run',json={'start':'generate','through':'pose'})
        self.assertEqual(response.status_code,400)
        self.assertEqual(read(self.directory/'job.json')['runs'],[])

    def test_missing_artifacts_block_dependent_stage(self):
        first=self.finish('trim','pose')
        Path(selected_run(first,'pose')['artifacts'][0]['pose_dir'],'sam3d_predictions.npz').unlink()
        self.assertIn('Saved pose',self.store.describe(self.job_id)['blocked']['generate'])

    def test_stage_route_and_model_urls_pin_training_attempt(self):
        first=self.finish('trim','splat')
        run=selected_run(first,'splat')
        output=Path(run['artifacts'][0]['training_dir'])
        (output/'scene.tsog').write_bytes(b'container1')
        write_json(output/'normalization.json', {'world_to_training':np.eye(4).tolist(),'training_to_world':np.eye(4).tolist()})
        info=self.client.get(f'/api/jobs/{self.job_id}/training?run={run["id"]}').json()
        self.assertIn('run='+run['id'],info['model'])
        self.assertEqual(self.client.get(info['model']).content,b'container1')
        self.finish('splat','splat',{'splat':{'steps':100}})
        self.assertEqual(self.client.get(info['model']).content,b'container1')
        self.assertEqual(self.client.get(f'/api/jobs/{self.job_id}/model?run=../../outside').status_code,404)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job_id}/stage?stage=pose&chunk=-1').json(),{'pending':True})

    def test_training_watch_exists_before_first_status_file(self):
        job=self.finish('trim','splat')
        run=selected_run(job,'splat')
        Path(run['artifacts'][0]['training_dir'],'status.json').unlink()
        stored=read(self.directory/'job.json')
        stored['runs'][-1]['status']='running';stored['active']={'stage':'splat','runs':[run['id']]}
        write_json(self.directory/'job.json',stored)
        self.manager.process=Mock();self.manager.process.poll.return_value=None
        with patch('fdanyone.space.server.export_result',return_value=({},[],{},'')):
            response=self.client.get(f'/api/jobs/{self.job_id}/stage?stage=splat&run={run["id"]}')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['training']['status'],'running')
        self.assertIn('run='+run['id'],response.json()['training']['watch'])
        self.manager.process=None

    def test_progressive_generation_uses_stage_scratch_and_version_cache(self):
        job=self.finish('trim','generate')
        run=selected_run(job,'generate')
        stored=read(self.directory/'job.json')
        result=self.directory/'stages'/run['id']/'0000/fdanyone/clip'
        stored['runs'][-1]['artifacts'][0]['result_dir']=str(result)
        stored['runs'][-1]['status']='running';stored['active']={'stage':'generate','runs':[run['id']]}
        write_json(self.directory/'job.json',stored)
        self.manager.process=Mock();self.manager.process.poll.return_value=None
        with patch('fdanyone.space.server.live_preview',return_value={'unchanged':True,'version':'body-ready'}) as preview:
            response=self.client.get(f'/api/jobs/{self.job_id}/stage?stage=generate&run={run["id"]}&version=body-ready')
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()['unchanged'])
        self.assertEqual(preview.call_args.args[0]['request'],str(result.parents[1]/'request.json'))
        self.assertEqual(preview.call_args.args[2],'body-ready')
        self.manager.process=None

    def test_adopt_legacy_result_keeps_files_and_reusable_training(self):
        legacy=self.manager.root/'job-abcdefabcdef'
        result=legacy/'results/0000/fdanyone/clip'
        (result/'training').mkdir(parents=True)
        clip=legacy/'clips/chunk_001.mp4';clip.parent.mkdir();clip.write_bytes(b'clip')
        write_json(legacy/'request.json',dict(video_paths=[str(self.manager.root/'uploads'/self.upload['id'])],
                   views_per_layer=9,layer_pitches=[0,30],turbo=True))
        request=result.parents[1]/'request.json'
        write_json(request,dict(video_path=str(clip),start_time=0))
        write_json(legacy/'ui.json',dict(source_names=['Original name.mp4']))
        write_json(legacy/'results/streams_report.json',dict(jobs=[dict(result_dir=str(result),request=str(request),status='completed')]))
        write_json(result/'metadata.json',dict(output=dict(fps='25/1')))
        write_json(result/'training/status.json',dict(status='completed',model='scene.tsog'))
        (result/'training/scene.tsog').write_bytes(b'original')
        write_json(result/'training/normalization.json',dict(world_to_training=np.eye(4).tolist(),training_to_world=np.eye(4).tolist()))
        response=self.client.get('/api/jobs/'+legacy.name)
        self.assertEqual(response.status_code,200,response.text)
        job=response.json()
        self.assertEqual(job['source']['name'],'Original name.mp4')
        self.assertEqual(job['configs']['generate']['views'],9)
        self.assertEqual(job['selected']['generate'],'generate-imported')
        self.assertIsNone(job['blocked']['splat'])
        model=self.client.get(f'/api/jobs/{legacy.name}/training?run=splat-imported').json()['model']
        self.assertEqual(self.client.get(model).content,b'original')
        self.assertEqual((result/'training/scene.tsog').read_bytes(),b'original')

    def test_second_server_shutdown_does_not_cancel_foreign_queue(self):
        self.finish('trim','pose')
        self.manager.process=None
        with patch.object(self.manager,'cancel') as cancel:
            self.manager.close()
        cancel.assert_not_called()

    def test_draft_deletion_preserves_upload(self):
        response=self.client.delete('/api/jobs/'+self.job_id)
        self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(self.directory.exists())
        self.assertTrue((self.manager.root/'uploads'/self.upload['id']).is_file())


if __name__=='__main__':
    unittest.main()
