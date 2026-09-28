"""Completed-job history and deletion stay inside the UI's owned directories."""
import json
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fdanyone.space.jobs import JobManager


class JobHistoryTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from fdanyone.space.server import create_app
        except ImportError:
            self.skipTest('Install the gui extra to test job history.')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        assets = self.root/'assets';assets.mkdir()
        self.app = create_app(self.root/'ui', web_assets=assets)
        self.manager = self.app.state.manager
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.upload = self.root/'ui/uploads/source.mp4';self.upload.write_bytes(b'source')
        self.model = self.root/'models/weights.bin';self.model.parent.mkdir();self.model.write_bytes(b'model')

    def job(self, number=1, *, chunks=1, status='completed', name='Walking.mp4'):
        directory = self.manager.root/f'job-{number:012x}'
        directory.mkdir()
        results = directory/'results';results.mkdir()
        plan = {'chunks': [{'label': f'Chunk {i+1}', 'start': i*4.84, 'end': (i+1)*4.84} for i in range(chunks)]}
        (directory/'request.json').write_text(json.dumps(dict(
            video_paths=[str(self.upload)], views_per_layer=6, layer_pitches=[0, 30], _clip_plan=plan)))
        (directory/'ui.json').write_text(json.dumps({'source_names': [name]}))
        jobs = []
        for i in range(chunks):
            result = results/f'chunk-{i}/fdanyone/result';result.mkdir(parents=True)
            (result/'video.mp4').write_bytes(b'generated video')
            log = results/f'chunk-{i}/worker.log';log.write_text('completed')
            jobs.append(dict(status=status, result_dir=str(result), video_path=str(self.upload), log=str(log)))
        report = results/'streams_report.json';report.write_text(json.dumps({'jobs':jobs}))
        os.utime(report, (1700000000+number, 1700000000+number))
        (directory/'clips').mkdir();(directory/'clips/chunk.mp4').write_bytes(b'prepared')
        preview = self.manager.root/'previews'/directory.name
        preview.mkdir();(preview/'browser.mp4').write_bytes(b'preview')
        return directory

    def test_history_is_persistent_named_and_sorted_with_chunk_results(self):
        first=self.job(1);second=self.job(2,chunks=2,name='Dance <take 2>.mp4')
        self.job(3,status='failed')
        history=self.client.get('/api/jobs').json()
        self.assertEqual([j['id'] for j in history],[second.name,first.name])
        self.assertEqual(history[0]['name'],'Dance <take 2>.mp4')
        self.assertEqual(history[0]['cameras'],12)
        self.assertEqual(history[0]['chunks'],2)
        self.assertEqual([r['label'] for r in history[0]['results']],['Chunk 1','Chunk 2'])
        self.assertGreater(history[0]['bytes'],len(b'generated video'))
        restored=JobManager(self.manager.root)
        self.assertEqual([j['id'] for j in restored.completed()],[second.name,first.name])

    def test_delete_current_job_removes_owned_data_and_restores_previous(self):
        previous=self.job(1);target=self.job(2,chunks=2)
        self.manager.directory=target
        external=self.root/'external-result';external.mkdir();(external/'keep.txt').write_text('keep')
        (target/'linked-external').symlink_to(external,target_is_directory=True)
        response=self.client.delete('/api/jobs/'+target.name)
        self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(target.exists())
        self.assertFalse((self.manager.root/'previews'/target.name).exists())
        for path in (previous,self.upload,self.model,external/'keep.txt',self.manager.root/'previews'/previous.name):
            self.assertTrue(path.exists(),str(path))
        report,_=self.manager.snapshot()
        self.assertEqual(report['directory'],str(previous))
        self.assertEqual(report['status'],'completed')
        self.assertEqual(len(self.client.get('/api/jobs').json()),1)
        self.assertEqual(len(self.client.get('/api/results').json()),1)
        self.assertEqual(self.client.delete('/api/jobs/'+target.name).status_code,400)
        self.assertEqual(self.client.delete('/api/jobs/'+previous.name).status_code,200)
        self.assertEqual(self.manager.snapshot()[0],{'status':'idle','jobs':[]})
        self.assertTrue(self.upload.exists());self.assertTrue(self.model.exists())

    def test_running_and_unfinished_jobs_cannot_be_deleted(self):
        active=self.job(1)
        self.manager.directory=active;self.manager.process=Mock()
        self.manager.process.poll.return_value=None
        try:
            self.assertEqual(self.client.get('/api/jobs').json(),[])
            self.assertEqual(self.client.delete('/api/jobs/'+active.name).status_code,400)
            self.assertTrue(active.exists())
        finally:
            self.manager.process=None
        unfinished=self.job(2,status='running')
        self.assertEqual(self.client.delete('/api/jobs/'+unfinished.name).status_code,400)
        self.assertTrue(unfinished.exists())

    def test_deleting_older_job_does_not_stop_another_running_job(self):
        active=self.job(1,status='running');older=self.job(2)
        self.manager.directory=active
        process=Mock();process.poll.return_value=None;self.manager.process=process
        try:
            self.assertEqual(self.client.delete('/api/jobs/'+older.name).status_code,200)
            self.assertIs(self.manager.process,process)
            self.assertEqual(self.manager.directory,active)
            self.assertEqual(self.manager.snapshot()[0]['status'],'running')
            self.assertTrue(active.exists())
            self.assertTrue((self.manager.root/'previews'/active.name).exists())
            process.terminate.assert_not_called();process.kill.assert_not_called()
        finally:
            self.manager.process=None

    def test_delete_rejects_external_ids_and_symlinked_jobs_or_caches(self):
        target=self.job(1)
        outside=self.root/'outside';outside.mkdir();(outside/'keep.txt').write_text('keep')
        alias=self.manager.root/'job-aaaaaaaaaaaa';alias.symlink_to(outside,target_is_directory=True)
        for job_id in ('models','..','job-aaaaaaaaaaaa','job-111111111111'):
            self.assertIn(self.client.delete('/api/jobs/'+job_id).status_code,(400,404,405))
        preview=self.manager.root/'previews'/target.name
        (preview/'browser.mp4').unlink();preview.rmdir();preview.symlink_to(outside,target_is_directory=True)
        self.assertEqual(self.client.delete('/api/jobs/'+target.name).status_code,400)
        self.assertTrue(target.exists());self.assertTrue((outside/'keep.txt').exists())

    def test_external_result_paths_are_not_offered_or_followed_by_delete(self):
        target=self.job(1)
        outside=self.root/'outside';outside.mkdir();(outside/'keep.txt').write_text('keep')
        report=target/'results/streams_report.json';data=json.loads(report.read_text())
        data['jobs'][0]['result_dir']=str(outside);report.write_text(json.dumps(data))
        self.assertEqual(self.client.get('/api/jobs').json()[0]['results'],[])
        self.assertEqual(self.client.delete('/api/jobs/'+target.name).status_code,200)
        self.assertTrue((outside/'keep.txt').exists())

    def test_cross_origin_delete_is_rejected(self):
        target=self.job(1)
        response=self.client.delete('/api/jobs/'+target.name,headers={'Origin':'https://elsewhere.example'})
        self.assertEqual(response.status_code,403);self.assertTrue(target.exists())

    def test_upload_name_is_saved_and_forwarded_to_job_history(self):
        upload=self.client.post('/api/uploads',files={'file':('Dance.mov',b'video')}).json()
        sidecar=self.manager.root/'uploads'/(upload['id']+'.upload.json')
        self.assertEqual(json.loads(sidecar.read_text())['name'],'Dance.mov')
        with patch.object(self.manager,'start',return_value='job') as start:
            response=self.client.post('/api/jobs',json={'videos':[upload['id']],'settings':{}})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(start.call_args.kwargs['source_names'],['Dance.mov'])

    def test_preview_namespace_matches_managed_job_and_external_results(self):
        target=self.job(1)
        self.assertEqual(self.manager.preview_root(target/'results/chunk-0'),self.manager.root/'previews'/target.name)
        self.assertEqual(self.manager.preview_root(self.root/'external'),self.manager.root/'previews/external')

    def thumbnail_result(self, number=1):
        from fractions import Fraction
        import numpy as np
        from fdanyone.video import write_video
        target = self.job(number)
        result = target/'results/chunk-0/fdanyone/result'
        colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]
        cameras = []
        for i, color in enumerate(colors):
            name = f'{i:02d}.mp4'
            frame = np.full((128, 64, 3), color, dtype=np.uint8)
            write_video((frame for _ in range(121)), result/name, Fraction(25), lossless_rgb=True)
            cameras.append(dict(camera_id=i, video=name, image_width=64, image_height=128,
                                K=np.eye(3).tolist(), camera_to_world=np.eye(4).tolist()))
        (result/'cameras.json').write_text(json.dumps(dict(camera_model='OPENCV',
            world_frame={'name':'canonical_human_world'}, camera_frame={'name':'opencv_camera'}, cameras=cameras)))
        (result/'metadata.json').write_text(json.dumps({'output':{'fps':'25/1','frames_per_video':121,'target_views':6}}))
        return target, result

    def test_gallery_thumbnail_has_three_angles_and_reuses_small_cache(self):
        from PIL import Image
        target, result = self.thumbnail_result()
        url = self.client.get('/api/jobs').json()[0]['results'][0]['thumbnail']
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, response.text if response.status_code != 200 else '')
        self.assertEqual(response.headers['content-type'], 'image/jpeg')
        image = Image.open(BytesIO(response.content))
        self.assertEqual(image.size, (480, 288))
        for x, expected in [(80,(255,0,0)), (240,(0,255,0)), (400,(0,0,255))]:
            self.assertTrue(all(abs(a-b)<5 for a,b in zip(image.getpixel((x,144)),expected)))
        cache = self.manager.root/'previews'/target.name
        self.assertEqual(len(list(cache.rglob('*.jpg'))), 1)
        self.assertEqual(list(cache.rglob('*.mp4')), [cache/'browser.mp4'])
        with patch('fdanyone.space.viewer.av.open', side_effect=AssertionError('Cached thumbnails must not decode again')):
            self.assertEqual(self.client.get(url).content, response.content)
        self.assertEqual(self.client.delete('/api/jobs/'+target.name).status_code, 200)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertFalse(cache.exists())

    def test_thumbnail_rejects_external_video_and_invalid_chunks(self):
        target, result = self.thumbnail_result()
        url = f'/api/jobs/{target.name}/thumbnail'
        for index in (-1, 1, 999):
            self.assertEqual(self.client.get(url, params={'index':index}).status_code, 404)
        (result/'00.mp4').unlink();(result/'00.mp4').symlink_to(self.upload)
        with patch('fdanyone.space.viewer.av.open') as decode:
            self.assertEqual(self.client.get(url).status_code, 404)
            decode.assert_not_called()
        self.assertEqual(self.client.get('/api/jobs/job-ffffffffffff/thumbnail').status_code, 404)

    def test_thumbnail_rejects_linked_cache_and_missing_video(self):
        target, result = self.thumbnail_result()
        url = f'/api/jobs/{target.name}/thumbnail'
        preview = self.manager.root/'previews'/target.name
        (preview/'gallery').symlink_to(self.root, target_is_directory=True)
        self.assertEqual(self.client.get(url).status_code, 404)
        (preview/'gallery').unlink()
        (result/'00.mp4').write_bytes(b'broken video')
        self.assertEqual(self.client.get(url).status_code, 404)


if __name__=='__main__':
    unittest.main()
