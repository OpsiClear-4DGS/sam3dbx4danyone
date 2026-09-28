"""4DGS export contracts: cameras, motion units, masking and failure retention."""
from fractions import Fraction
import importlib.util
import json
import os
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from fdanyone.errors import ConfigurationError
from fdanyone.io import write_json
from fdanyone.reconstruction import REPO, worker_environment
from fdanyone.reconstruction.dataset import mesh_initialization, prepare_dataset, read_result
from fdanyone.video import write_video


def fixture(root):
    root.mkdir()
    vertices = np.array([[-.3,-.3,0], [.3,-.3,0], [0,.3,0], [0,0,.5]], dtype=np.float32)
    sequence = np.stack([vertices + [i*.01, 0, 0] for i in range(7)]).astype(np.float32)
    faces = np.array([[0,2,1], [0,1,3], [1,2,3], [2,0,3]])
    (root/'preprocessing').mkdir()
    np.savez(root/'preprocessing/viewer_geometry.npz', vertices=sequence, faces=faces)
    cameras = []
    for index in range(2):
        c2w = np.eye(4)
        c2w[:3, 3] = [index*.1, 0, -3]
        path = f'videos/dense/{index:02d}.mp4'
        write_video((np.full((48,32,3), 100+index*20, np.uint8) for _ in range(7)),
                    root/path, Fraction(25), lossless_rgb=True)
        cameras.append(dict(camera_id=index, image_width=32, image_height=48,
                            K=[[30,0,16],[0,30,24],[0,0,1]], camera_to_world=c2w.tolist(), video=path))
    write_json(root/'cameras.json', dict(camera_model='OPENCV', cameras=cameras))
    write_json(root/'metadata.json', dict(output=dict(frames_per_video=7, width=32,height=48,fps='25/1')))
    return sequence, faces, cameras


class ExportTests(unittest.TestCase):
    def test_mesh_seeds_preserve_motion_and_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            vertices, faces, _ = fixture(Path(temporary)/'input')
            keys, positions, velocities, normals = mesh_initialization(vertices, faces, samples=32)
            self.assertEqual(keys, [0,5,6])
            np.testing.assert_allclose(velocities, np.broadcast_to([.01,0,0], velocities.shape), atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=2), 1, atol=1e-6)

    def test_full_export_preserves_cameras_time_and_straight_rgba(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, cameras = fixture(root/'input')
            def segment(frames):
                mask = np.full((len(frames),48,32), 255, np.uint8)
                mask[:, :10] = 0
                mask[:, 10:20] = 128
                return mask
            report = prepare_dataset(root/'input', root/'dataset', segment, samples=32)
            self.assertEqual(report['initial_gaussians'], 96)
            self.assertEqual(len(list((root/'dataset/images').rglob('*.png'))), 14)
            image = np.asarray(Image.open(root/'dataset/images/cam000/000000.png'))
            self.assertEqual(image.shape, (48,32,4))
            self.assertTrue((image[..., :3] == 100).all())
            self.assertTrue((image[:10, :, 3] == 0).all())
            self.assertTrue((image[10:20, :, 3] == 128).all())
            self.assertTrue((image[20:, :, 3] == 255).all())
            self.assertEqual(report['alpha_mode'], 'transparent')
            self.assertEqual(report['format_version'], 2)
            with (root/'dataset/sparse/0/images.bin').open('rb') as stream:
                self.assertEqual(struct.unpack('<Q', stream.read(8))[0],2)
                for index in range(2):
                    fields = struct.unpack('<i7di', stream.read(64))
                    pose = np.asarray(fields[1:8], float)
                    rotation = Rotation.from_quat([*pose[1:4], pose[0]]).as_matrix()
                    w2c = np.eye(4); w2c[:3,:3] = rotation; w2c[:3,3] = pose[4:]
                    np.testing.assert_allclose(np.linalg.inv(w2c), cameras[index]['camera_to_world'], atol=1e-7)
                    name = b''.join(iter(lambda:stream.read(1),b'\0')).decode()
                    self.assertEqual(name, f'cam{index:03d}/000000.png')
                    self.assertEqual(struct.unpack('<Q', stream.read(8))[0],0)
            with np.load(root/'dataset/init.npz') as data:
                np.testing.assert_allclose(np.unique(data['times']), [0,5/6,1])
                self.assertEqual(int(data['frame_end']),7)
                self.assertEqual(int(data['time_denominator']),6)
            loader = REPO/'src/init_common.py'
            if loader.exists():
                spec = importlib.util.spec_from_file_location('ftgs_init_test', loader)
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                values = module.read_init_arrays(str(root/'dataset/init.npz'),0,7)
                np.testing.assert_allclose(values['velocities'], np.tile([.06,0,0],(96,1)), atol=1e-6)

    def test_legacy_composited_dataset_requires_regeneration(self):
        from fdanyone.reconstruction.train_worker import validate_dataset_manifest
        with self.assertRaisesRegex(ValueError, 'original videos'):
            validate_dataset_manifest({'image_format':'lossless PNG'})
        validate_dataset_manifest({'format_version':2, 'alpha_mode':'transparent'})

    def test_invalid_frames_do_not_publish_a_partial_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root/'input')
            write_video((np.full((48,32,3), 120, np.uint8) for _ in range(6)),
                        root/'input/videos/dense/00.mp4', Fraction(25), lossless_rgb=True)
            with self.assertRaisesRegex(ConfigurationError, 'expected 7'):
                prepare_dataset(root/'input',root/'dataset',lambda batch:np.full((len(batch),48,32),255,np.uint8),samples=4)
            self.assertFalse((root/'dataset').exists())
            self.assertFalse(list(root.glob('.dataset.work-*')))

    def test_external_video_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root/'input')
            video = root/'input/videos/dense/00.mp4'
            video.rename(root/'outside.mp4')
            video.symlink_to(root/'outside.mp4')
            with self.assertRaisesRegex(ConfigurationError, 'inside'):
                read_result(root/'input')

    def test_trainer_environment_keeps_gpu_uuid_without_inference_allocator(self):
        with patch.dict(os.environ, {'PYTHONPATH':'generation', 'VIRTUAL_ENV':'generation',
                                    'PYTORCH_ALLOC_CONF':'expandable_segments:True'}):
            environment = worker_environment('GPU-training')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'],'GPU-training')
        for name in ('PYTHONPATH','VIRTUAL_ENV','PYTORCH_ALLOC_CONF'):
            self.assertNotIn(name, environment)

    def test_missing_runtime_fails_before_generating(self):
        from inference import inference
        with (patch('fdanyone.reconstruction.check_runtime', side_effect=ConfigurationError('setup')),
              patch('fdanyone.pipeline.run_pipeline') as generation):
            with self.assertRaisesRegex(ConfigurationError,'setup'):
                inference('anything.mp4', train_4dgs=True)
            generation.assert_not_called()

    def test_training_follows_generation_on_selected_gpu(self):
        from inference import inference
        with (patch('fdanyone.reconstruction.check_runtime'),
              patch('fdanyone.pipeline.run_pipeline', return_value={'result_dir':'saved'}) as generate,
              patch('fdanyone.reconstruction.train_result', return_value={'status':'completed'}) as train):
            result = inference('source.mp4', gpu_ids=[3], train_4dgs=True, training_steps=200)
        self.assertEqual(result['training']['status'],'completed')
        generate.assert_called_once()
        train.assert_called_once_with('saved', model_dir='models', gpu_id=3, steps=200)


class TrainingAPITests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from fdanyone.space.server import create_app
        except ImportError:
            self.skipTest('Install the gui extra to test the HTTP API.')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        assets = root/'assets'; assets.mkdir()
        self.app = create_app(root/'ui', web_assets=assets)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.job = self.app.state.manager.root/'job-012345abcdef'
        self.result = self.job/'results/0000/fdanyone/clip'
        (self.result/'training').mkdir(parents=True)
        write_json(self.job/'request.json', {'video_paths':['clip.mp4'], 'views_per_layer':6,'layer_pitches':[15]})
        write_json(self.result/'metadata.json', {'output':{'fps':'25/1'}})
        self.report = {'jobs':[{'status':'completed', 'result_dir':str(self.result),'log':str(self.job/'worker.log')}]}
        write_json(self.job/'results/streams_report.json', self.report)
        write_json(self.result/'training/status.json', {'status':'completed','stage':'completed','steps':30000,'step':30000})
        (self.result/'training/scene.ftgs.ply').write_bytes(b'ply\nmodel data')
        write_json(self.result/'training/normalization.json', {
            'world_to_training':np.eye(4).tolist(),'training_to_world':np.eye(4).tolist()})

    def test_trained_model_includes_viewer_transform_and_range_download(self):
        with patch('fdanyone.space.server.export_result', return_value=({},[],{'output':{'fps':'25/1'}},'')):
            response = self.client.post('/api/results/open',json={'directory':str(self.result)})
        self.assertEqual(response.status_code,200,response.text)
        training = response.json()['training']
        self.assertNotIn('player',training)
        self.assertFalse(training['live'])
        np.testing.assert_allclose(training['training_to_world'],np.eye(4))
        response = self.client.get(training['model'],headers={'Range':'bytes=0-3'})
        self.assertEqual(response.status_code,206)
        self.assertEqual(response.content,b'ply\n')
        if (REPO/'player/index.html').is_file():
            response = self.client.get('/ftgs/ftgs.js')
            self.assertIn('javascript',response.headers['content-type'])
            self.assertIn('readFTGS',response.text)
            self.assertEqual(self.client.get('/ftgs/sort.js').status_code,200)
            self.assertEqual(self.client.get('/ftgs/app.js').status_code,404)

    def test_live_snapshot_is_playable_before_training_or_batch_completes(self):
        self.report['jobs'][0]['status']='running'
        write_json(self.job/'results/streams_report.json',self.report)
        output=self.result/'training'
        preview=output/'previews/step-000000101.ftgs.ply'
        preview.parent.mkdir();preview.write_bytes(b'ply\nlive snapshot')
        write_json(output/'status.json',dict(status='running',stage='training',step=251,steps=30000,preview_step=101))
        response=self.client.get('/api/jobs/job-012345abcdef/training')
        self.assertEqual(response.status_code,200,response.text)
        info=response.json()
        self.assertTrue(info['live']);self.assertEqual(info['preview_step'],101)
        self.assertEqual(self.client.get(info['model']).content,b'ply\nlive snapshot')
        self.assertEqual(self.client.get('/api/jobs/job-012345abcdef/model').status_code,404)
        self.assertEqual(self.client.get('/api/jobs/job-012345abcdef/model?preview=102').status_code,404)
        self.assertEqual(self.client.get('/api/jobs/job-012345abcdef/model?preview=-1').status_code,404)
        # Unchanged generation videos must still carry new training state.
        from unittest.mock import Mock
        manager=self.app.state.manager;manager.directory=self.job
        manager.process=Mock();manager.process.poll.return_value=None
        self.report['jobs'][0]['request']=str(self.job/'request.json')
        write_json(self.job/'results/streams_report.json',self.report)
        with patch('fdanyone.space.server.live_preview',return_value={'unchanged':True,'version':'generated'}):
            result=self.client.get('/api/jobs/preview?version=generated').json()
        self.assertEqual(result['training']['preview_step'],101)
        manager.process=None

    def test_live_snapshot_and_normalization_cannot_escape_owned_result(self):
        output=self.result/'training'
        preview=output/'previews/step-000000101.ftgs.ply'
        preview.parent.mkdir();preview.symlink_to(self.job.parent/'private.ply')
        (self.job.parent/'private.ply').write_bytes(b'private')
        write_json(output/'status.json',dict(status='running',preview_step=101))
        self.assertEqual(self.client.get('/api/jobs/job-012345abcdef/model?preview=101').status_code,404)
        self.assertIsNone(self.client.get('/api/jobs/job-012345abcdef/training').json()['model'])
        preview.unlink();preview.write_bytes(b'ply')
        normalization=output/'normalization.json';normalization.unlink()
        normalization.symlink_to(self.job.parent/'private.ply')
        self.assertIsNone(self.client.get('/api/jobs/job-012345abcdef/training').json()['model'])

    def test_model_route_rejects_symlink_escape_and_unfinished_output(self):
        model = self.result/'training/scene.ftgs.ply'
        model.unlink()
        outside = self.job.parent/'private.ply';outside.write_bytes(b'private')
        model.symlink_to(outside)
        self.assertEqual(self.client.get('/api/jobs/job-012345abcdef/model').status_code,404)
        write_json(self.result/'training/status.json',{'status':'running'})
        self.assertEqual(self.client.get('/api/jobs/job-012345abcdef/model').status_code,404)

    def test_completed_chunk_is_playable_while_another_chunk_trains(self):
        from unittest.mock import Mock
        manager = self.app.state.manager
        manager.directory = self.job
        manager.process = Mock(); manager.process.poll.return_value=None
        self.report['jobs'].append({'status':'running','result_dir':str(self.job/'pending')})
        write_json(self.job/'results/streams_report.json',self.report)
        self.assertEqual(self.client.get('/api/jobs').json(),[])
        with patch('fdanyone.space.server.export_result', return_value=({},[],{'output':{'fps':'25/1'}},'')):
            result=self.client.post('/api/results/open',json={'directory':str(self.result)}).json()
        self.assertIsNotNone(result['training'])
        self.assertEqual(self.client.get(result['training']['model']).status_code,200)
        manager.process = None

    def test_live_views_survive_generation_scratch_cleanup(self):
        from fdanyone.space.viewer import live_preview
        job = dict(result_dir=str(self.result), request=str(self.job/'request.json'))
        with patch('fdanyone.space.viewer.export_result', return_value=({'frames':121},['video'],{},'')) as export:
            response = live_preview(job,self.job/'previews')
            self.assertEqual(response['scene']['frames'],121)
            unchanged = live_preview(job,self.job/'previews',response['version'])
            self.assertTrue(unchanged['unchanged'])
            export.assert_called_once()

    def test_failed_training_keeps_generated_results_in_gallery(self):
        self.report['jobs'][0]['status']='failed'
        write_json(self.job/'results/streams_report.json',self.report)
        write_json(self.result/'training/status.json',{'status':'failed'})
        self.assertEqual(self.client.get('/api/jobs').json()[0]['results'][0]['training'],'failed')
        self.assertIn(str(self.result),self.client.get('/api/results').json())
        # A failed generation without a published result must still be hidden.
        (self.result/'metadata.json').unlink()
        self.assertEqual(self.client.get('/api/jobs').json(),[])

    def test_submission_carries_training_choice_and_status_exposes_stage(self):
        token = self.client.post('/api/uploads', files={'file':('clip.mp4',b'video')}).json()['id']
        with patch.object(self.app.state.manager,'start',return_value='job') as start:
            response = self.client.post('/api/jobs',json={'videos':[token],'settings':{'train_4dgs':True}})
            self.assertEqual(response.status_code,200,response.text)
            self.assertTrue(start.call_args.args[1]['train_4dgs'])
        from unittest.mock import Mock
        manager = self.app.state.manager
        manager.directory = self.job
        manager.process = Mock();manager.process.poll.return_value=None
        self.report['jobs'][0]['status']='running'
        write_json(self.job/'results/streams_report.json',self.report)
        write_json(self.result/'training/status.json',{'status':'running','stage':'training','step':500,'steps':30000})
        response = self.client.get('/api/status').json()['report']
        self.assertEqual(response['stage'],'training')
        self.assertEqual(response['jobs'][0]['training']['step'],500)
        self.assertEqual(self.client.get('/api/jobs').json(),[])
        manager.process = None


if __name__ == '__main__':
    unittest.main()
