"""GUI adapters retain the pipeline's camera, queue and output contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fdanyone.space.jobs import JobManager, make_options
from fdanyone.space.viewer import contained_file


class UIContractTests(unittest.TestCase):
    def test_turbo_rejects_unsupported_layout(self):
        for views, pitches in [(20, '[15]'), (6, '[-15,0,15,30]')]:
            with self.subTest(views=views, pitches=pitches), self.assertRaises(ValueError):
                make_options(views, pitches, 0, 360, True, 0, 'auto', 42)
        options = make_options(16, '[-10,15,35]', 0, 360, False, 0, 'auto', 42)
        self.assertFalse(options['turbo'])
        self.assertTrue(options['compile_dit'])
        self.assertEqual(options['execution_profile'], 'full')

    def test_invalid_clip_settings(self):
        for start, fps in [(-1, 'auto'), (float('nan'), 'auto'), (0, '0')]:
            with self.subTest(start=start, fps=fps), self.assertRaises(ValueError):
                make_options(6, '[15]', 0, 360, True, start, fps, 42)

    def test_output_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'result';root.mkdir()
            external = Path(d)/'outside.mp4';external.touch()
            (root/'video.mp4').symlink_to(external)
            with self.assertRaises(ValueError):
                contained_file(root, 'video.mp4')

    def test_batch_submission_and_running_guard(self):
        with tempfile.TemporaryDirectory() as d, patch('fdanyone.space.jobs.subprocess.Popen') as popen:
            popen.return_value.poll.return_value = None
            source = Path(d)/'source.mp4';source.touch()
            manager = JobManager(Path(d)/'ui')
            opts = make_options(24, '[15]', 0, 360, True, 0, 'auto', 42)
            directory = Path(manager.start([source], opts))
            request = json.loads((directory/'request.json').read_text())
            self.assertEqual(request['video_paths'], [str(source)])
            self.assertNotIn('gpu_ids', request)
            self.assertTrue(request['turbo'])
            with self.assertRaises(ValueError):
                manager.start([source], opts)

    def test_cancel_signals_launcher_for_queue_cleanup(self):
        with tempfile.TemporaryDirectory() as d, patch('fdanyone.space.jobs.os.killpg') as kill:
            manager = JobManager(d)
            from unittest.mock import Mock
            manager.process = Mock(pid=123)
            manager.process.poll.return_value = None
            manager.cancel()
            kill.assert_called_once()
            manager.process.wait.assert_called_once_with(timeout=30)


class WebAPITests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from fdanyone.space.server import create_app
        except ImportError:
            self.skipTest('Install the gui extra to test the HTTP API.')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        assets = self.root/'assets';assets.mkdir()
        self.app = create_app(self.root/'ui', web_assets=assets)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_upload_and_submit_uses_full_queue_contract(self):
        response = self.client.post('/api/uploads', files={'file': ('source.mp4', b'video', 'video/mp4')})
        self.assertEqual(response.status_code, 200)
        token = response.json()['id']
        with patch.object(self.app.state.manager, 'start', return_value='job') as start:
            response = self.client.post('/api/jobs', json={'videos': [token], 'settings': {}})
            self.assertEqual(response.status_code, 200)
            paths, options = start.call_args.args
            self.assertTrue(Path(paths[0]).is_file())
            self.assertEqual(options['execution_profile'], 'full')
            self.assertTrue(options['turbo'])
            self.assertTrue(options['compile_dit'])

    def test_clip_workflow_plan_and_job_submission(self):
        from fractions import Fraction
        import numpy as np
        from fdanyone.video import write_video
        source = self.root/'source.mp4'
        write_video((np.full((16,24,3),i%256,dtype=np.uint8) for i in range(500)),
                    source, Fraction(25), lossless_rgb=True)
        upload = self.client.post('/api/uploads', files={'file':('source.mp4', source.read_bytes())}).json()
        token = upload['id']
        info = self.client.get(f'/api/uploads/{token}/info').json()
        self.assertEqual(info['frames'], 500)
        self.assertNotIn('timestamps', info)
        timeline = self.client.get(f'/api/uploads/{token}/info?timeline=true').json()
        self.assertEqual(timeline['chunk_frames'], 121)
        self.assertEqual(timeline['timestamps'], [i/25 for i in range(500)])
        self.assertNotIn('identity', timeline)
        edit = {'start':.4,'end':19.8,'skip':2}
        preview = self.client.post('/api/clips/plan', json={'video':token, **edit})
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(len(preview.json()['chunks']), 2)
        with patch.object(self.app.state.manager, 'start', return_value='job') as start:
            response = self.client.post('/api/jobs', json={
                'videos':[token], 'settings':{'views':9,'pitches':[0,30]}, 'clip':edit})
            self.assertEqual(response.status_code, 200)
            plan = start.call_args.kwargs['clip_plan']
            self.assertEqual(plan['kept_frames'],162)
            self.assertEqual(start.call_args.args[1]['views_per_layer'],9)
        self.assertEqual(self.client.post('/api/clips/plan',json={'video':token,'start':5,'end':2}).status_code,400)
        self.assertEqual(self.client.post('/api/clips/plan',json={'video':token,'skip':True}).status_code,422)

    def test_cross_origin_mutation_and_external_media_rejected(self):
        response = self.client.post('/api/cancel', headers={'Origin': 'https://unrelated.example'})
        self.assertEqual(response.status_code, 403)
        external = self.root/'secret.txt';external.write_text('private')
        (self.root/'ui/uploads/leak.txt').symlink_to(external)
        self.assertEqual(self.client.get('/media/uploads/leak.txt').status_code, 404)

    def test_invalid_upload_and_settings_rejected(self):
        self.assertEqual(self.client.post('/api/uploads', files={'file': ('script.py', b'x')}).status_code, 400)
        self.assertEqual(self.client.post('/api/layout', json={'views': 20, 'turbo': True}).status_code, 400)
        self.assertEqual(self.client.post('/api/layout', json={'views': 6, 'shell': 'anything'}).status_code, 422)

    def test_media_range_requests(self):
        p = self.root/'ui/previews/test.mp4';p.write_bytes(b'0123456789')
        response = self.client.get('/media/previews/test.mp4', headers={'Range':'bytes=2-5'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, b'2345')

    def test_three_module_is_served_locally(self):
        module = self.root/'assets/build/three.module.js'
        module.parent.mkdir(); module.write_text('export const ready = true;')
        response = self.client.get('/three/build/three.module.js')
        self.assertEqual(response.status_code, 200)
        self.assertIn('javascript', response.headers['content-type'])

    def test_layout_contains_calibrated_cameras_without_recording(self):
        response = self.client.post('/api/layout', json={'views': 24})
        self.assertEqual(response.status_code, 200)
        scene = response.json()['scene']
        self.assertEqual(len(scene['cameras']), 24)
        self.assertEqual(scene['frames'], 1)
        self.assertEqual(len(scene['cameras'][0]['camera_to_world']), 4)
        self.assertNotIn('recording', response.json())

    def test_all_ring_presets_preview_exact_total(self):
        for total, rings in ((6,1),(8,1),(12,1),(16,1),(16,2),
                             (18,1),(18,2),(24,2),(24,3),(36,3)):
            pitches = {1:[15], 2:[0,30], 3:[-15,15,45]}[rings]
            for turbo in (True, False):
                with self.subTest(total=total, rings=rings, turbo=turbo):
                    response = self.client.post('/api/layout', json={
                        'views':total//rings, 'pitches':pitches, 'turbo':turbo})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(len(response.json()['scene']['cameras']), total)

    def test_no_active_live_preview(self):
        self.assertEqual(self.client.get('/api/jobs/preview').json(), {'pending': True})
        self.assertEqual(self.client.get('/api/jobs/preview?index=-1').json(), {'pending': True})

    def test_mesh_url_is_local_media_with_range_support(self):
        mesh = self.root/'ui/previews/body.bin';mesh.write_bytes(b'MHR1' + b'0'*12)
        scene = {'mesh': {'url': str(mesh)}}
        with patch('fdanyone.space.server.export_result', return_value=(scene, [], {}, '')):
            result = self.client.post('/api/results/open', json={'directory': 'result'}).json()
        url = result['scene']['mesh']['url']
        self.assertEqual(url, '/media/previews/body.bin')
        response = self.client.get(url, headers={'Range': 'bytes=0-3'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, b'MHR1')

class ThreeSceneTests(unittest.TestCase):
    def test_mesh_export_preserves_animation_and_topology(self):
        import numpy as np
        import struct
        from fdanyone.space.viewer import export_mesh
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);geometry = root/'geometry.npz'
            points = np.zeros((121, 70, 3), dtype=np.float32)
            np.savez_compressed(geometry, keypoints=points)
            self.assertIsNone(export_mesh(geometry, root))
            vertices = np.arange(121*4*3, dtype=np.float32).reshape(121, 4, 3) / 100
            faces = np.array([[0, 1, 2], [1, 2, 3]])
            np.savez_compressed(geometry, keypoints=points, vertices=vertices, faces=faces)
            result = export_mesh(geometry, root)
            data = Path(result['url']).read_bytes()
            self.assertEqual(struct.unpack('<4sIII', data[:16]), (b'MHR1', 121, 4, 2))
            np.testing.assert_array_equal(np.frombuffer(data, '<f4', vertices.size, 16).reshape(vertices.shape), vertices)
            np.testing.assert_array_equal(np.frombuffer(data, '<u4', faces.size, 16+vertices.nbytes).reshape(faces.shape), faces)
            self.assertEqual(export_mesh(geometry, root), result)
            faces[0, 0] = 4
            np.savez_compressed(geometry, keypoints=points, vertices=vertices, faces=faces)
            with self.assertRaises(ValueError):
                export_mesh(geometry, root)
            faces[0, 0] = 0;vertices[0, 0, 0] = np.nan
            np.savez_compressed(geometry, keypoints=points, vertices=vertices, faces=faces)
            with self.assertRaises(ValueError):
                export_mesh(geometry, root)

    def test_body_geometry_keeps_translation_and_camera_coordinates(self):
        import numpy as np
        import torch
        from unittest.mock import Mock
        from fdanyone.motion.conditioning import body_geometry, save_viewer_geometry
        model = Mock()
        v = torch.tensor([[[-50., -100., 0.], [50., -100., 0.], [0., 100., 0.]]])
        model.side_effect = lambda shape, params, expr: (v.expand(len(shape), -1, -1), v.expand(len(shape), -1, -1))
        model.character_torch.mesh.faces = torch.tensor([[0, 1, 2]])
        points = np.zeros((121, 70, 3), dtype=np.float32)
        points[:, [5, 9], 0] = .5;points[:, [6, 10], 0] = -.5
        translation = np.zeros((121, 3), dtype=np.float32)
        translation[:, 0] = np.arange(121) * .01;translation[:, 2] = 6
        predictions = dict(pred_keypoints_3d=points, pred_cam_t=translation,
                           shape=np.zeros((121, 45), dtype=np.float32), mhr_model_params=np.zeros((121, 204), dtype=np.float32))
        with patch('torch.jit.load', return_value=model):
            geometry, faces = body_geometry(predictions, 'unused.pt', 'cpu')
        self.assertAlmostEqual(float(geometry.vertices_world[..., 1].min()), 0)
        np.testing.assert_allclose(geometry.vertices_world[-1] - geometry.vertices_world[0], [[1.2, 0, 0]]*3, atol=1e-6)
        transform = geometry.motion_world_to_canonical_world
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1)
        world = (points + translation[:, None]) * [1, -1, -1]
        expected = world @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(geometry.keypoints_world, expected, atol=1e-6)
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'viewer_geometry.npz';save_viewer_geometry(target, geometry, faces)
            with np.load(target, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['vertices'], geometry.vertices_world)
                np.testing.assert_array_equal(data['faces'], faces)

    def test_skeleton_contract_and_colors(self):
        import numpy as np
        from fdanyone.space.viewer import scene_payload
        points = np.zeros((121,70,3))
        scene = scene_payload([],25,points)
        self.assertEqual(scene['frames'],121)
        self.assertEqual(len(scene['links'][0]['color']),3)
        points[0,0,0] = np.nan
        with self.assertRaises(ValueError):
            scene_payload([],25,points)

    def test_live_preview_waits_for_complete_video_and_caches_version(self):
        import numpy as np
        from fractions import Fraction
        from fdanyone.space.viewer import live_preview
        from fdanyone.video import write_video
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scratch=root/'.sam3d-test';prep=scratch/'preprocessing'
            conditioning=prep/'conditioning';conditioning.mkdir(parents=True)
            camera=dict(camera_id=0,K=[[25,0,8],[0,25,16],[0,0,1]],
                        camera_to_world=np.eye(4).tolist(),image_width=16,image_height=32)
            (conditioning/'cameras.json').write_text(json.dumps({'cameras':[camera]}))
            (conditioning/'metadata.json').write_text(json.dumps({'fps_num':25,'fps_den':1}))
            np.savez_compressed(prep/'viewer_geometry.npz',keypoints=np.zeros((121,70,3)),
                                vertices=np.zeros((121,3,3)),faces=np.array([[0,1,2]]))
            job={'request':str(root/'request.json')}
            first=live_preview(job,root/'cache')
            self.assertEqual(first['videos'],[None]);self.assertEqual(first['scene']['frames'],121)
            self.assertTrue(Path(first['scene']['mesh']['url']).is_file())
            self.assertTrue(live_preview(job,root/'cache',first['version'])['unchanged'])
            source=scratch/'generation/target/videos/00.mp4';source.parent.mkdir(parents=True)
            source.write_bytes(b'incomplete mp4')
            self.assertTrue(live_preview(job,root/'cache',first['version'])['unchanged'])
            write_video((np.zeros((32,16,3),dtype=np.uint8) for _ in range(121)),source,Fraction(25))
            finished=live_preview(job,root/'cache',first['version'])
            self.assertNotEqual(finished['version'],first['version'])
            self.assertTrue(Path(finished['videos'][0]).is_file())

    def test_completed_queue_is_restored_after_ui_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=root/'job-test/results/streams_report.json';report.parent.mkdir(parents=True)
            report.write_text(json.dumps({'jobs':[{'status':'completed','log':str(root/'worker.log')}]}))
            manager=JobManager(root)
            data,_=manager.snapshot()
            self.assertEqual(data['status'],'completed')


if __name__ == '__main__':
    unittest.main()
