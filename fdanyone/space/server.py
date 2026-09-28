"""HTTP API and static web interface for the existing GPU queue."""
from contextlib import asynccontextmanager
import json
import av
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.parse import quote
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from fdanyone.config import INFERENCE
from fdanyone.errors import FourDAnyoneError
from fdanyone.io import write_json
from fdanyone.space.jobs import JobManager, make_options, has_generated_result
from fdanyone.space.clips import video_index, plan_clip
from fdanyone.space.viewer import export_layout, export_result, live_preview, result_thumbnail
from fdanyone.space.web_assets import prepare_web_assets

STATIC = Path(__file__).with_name('static')
MAX_UPLOAD = 2 * 1024**3
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    views: int = 6
    pitches: list[int] = Field(default_factory=lambda: [15])
    yaw: int = 0
    span: int = 360
    turbo: bool = True
    start_time: float = 0
    fps: str = 'auto'
    seed: int = 42
    train_4dgs: bool = False

    def options(self):
        return dict(make_options(self.views, self.pitches, self.yaw, self.span,
                                 self.turbo, self.start_time, self.fps, self.seed),
                    train_4dgs=self.train_4dgs)


class ClipEdit(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    start: float = Field(default=0, ge=0)
    end: float | None = Field(default=None, gt=0)
    skip: int = Field(default=0, ge=0, le=15, strict=True)


class ClipPlanRequest(ClipEdit):
    video: str


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    videos: list[str] = Field(min_length=1)
    settings: Settings
    clip: ClipEdit | None = None


class ResultRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    directory: str = Field(min_length=1)


def within(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise HTTPException(404, 'File not found.')
    return path


def create_app(cache_dir='outputs/ui', model_dir='models', *, video_path=None,
               output_dir=None, web_assets=None):
    manager = JobManager(cache_dir, model_dir)
    root = manager.root
    uploads = root/'uploads'
    previews = root/'previews'
    uploads.mkdir(exist_ok=True)
    previews.mkdir(exist_ok=True)
    assets = Path(web_assets) if web_assets else prepare_web_assets(root/'assets')
    artifact_lock = threading.RLock()
    initial = []
    if video_path:
        source = Path(video_path).expanduser().resolve(strict=True)
        token = uuid.uuid4().hex + source.suffix.lower()
        shutil.copyfile(source, uploads/token)
        write_json(uploads/(token+'.upload.json'), {'name': source.name})
        initial.append({'id': token, 'name': source.name, 'url': '/media/uploads/'+token})

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title='sam3dbx4danyone', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.manager = manager

    @app.middleware('http')
    async def same_origin(request, call_next):
        origin = request.headers.get('origin')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin requests are not permitted.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.exception_handler(FourDAnyoneError)
    @app.exception_handler(ValueError)
    def bad_request(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    @app.exception_handler(av.error.FFmpegError)
    def invalid_video(request, exc):
        return JSONResponse({'detail': 'Cannot decode this video. Please upload a valid video file.'}, status_code=400)

    @app.exception_handler(subprocess.CalledProcessError)
    def conversion_failed(request, exc):
        return JSONResponse({'detail': 'Video conversion failed. Check the input video and server log.'}, status_code=400)

    @app.get('/')
    def index():
        return FileResponse(STATIC/'index.html', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/config')
    def config():
        from fdanyone.reconstruction import check_runtime
        try:
            check_runtime()
            training_available = True
        except FourDAnyoneError:
            training_available = False
        return {'videos': initial, 'output_dir': str(output_dir) if output_dir else '',
                'max_upload_bytes': MAX_UPLOAD, 'training_available': training_available}

    @app.post('/api/uploads')
    async def upload(file: UploadFile):
        name = Path((file.filename or 'video.mp4').replace('\\', '/')).name
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            await file.close()
            raise HTTPException(400, 'Choose MP4, MOV, MKV, WebM, AVI or M4V video.')
        token = uuid.uuid4().hex + suffix
        target = uploads/token
        temporary = target.with_suffix(suffix+'.part')
        try:
            size = 0
            with temporary.open('wb') as stream:
                while chunk := await file.read(1024*1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'Each upload must be at most 2 GiB.')
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, 'The uploaded file is empty.')
            temporary.replace(target)
            write_json(uploads/(token+'.upload.json'), {'name': name})
            return {'id': token, 'name': name, 'url': '/media/uploads/'+token}
        finally:
            temporary.unlink(missing_ok=True)
            await file.close()

    @app.get('/api/uploads/{token}/info')
    def upload_info(token: str, timeline: bool = False):
        data = video_index(within(uploads, token))
        info = {k:v for k,v in data.items() if k not in ('timestamps','identity')}
        if timeline:
            info.update(timestamps=data['timestamps'], chunk_frames=INFERENCE.num_frames)
        return info

    @app.post('/api/clips/plan')
    def clip_plan(body: ClipPlanRequest):
        plan = plan_clip(within(uploads, body.video), body.start, body.end, body.skip)
        return {k:v for k,v in plan.items() if k not in ('source','identity')}

    @app.post('/api/jobs')
    def submit(body: Submission):
        paths = [str(within(uploads, name)) for name in body.videos]
        names = []
        for path in paths:
            path = Path(path)
            sidecar = path.with_suffix(path.suffix+'.upload.json')
            names.append(json.loads(within(uploads, sidecar.name).read_text())['name'] if sidecar.exists() else path.name)
        if body.clip is not None:
            if len(paths) != 1:
                raise ValueError('Trim and split accepts one source video per batch.')
            plan = plan_clip(paths[0], body.clip.start, body.clip.end, body.clip.skip)
            directory = manager.start(paths, body.settings.options(), clip_plan=plan, source_names=names)
        else:
            directory = manager.start(paths, body.settings.options(), source_names=names)
        return {'directory': directory}

    @app.get('/api/jobs')
    def completed_jobs():
        jobs = manager.completed()
        for job in jobs:
            for index, result in enumerate(job['results']):
                result['thumbnail'] = f"/api/jobs/{job['id']}/thumbnail?index={index}&v={job['completed_at']}"
        return jobs

    @app.get('/api/jobs/{job_id}/thumbnail')
    def thumbnail(job_id: str, index: int = 0):
        with artifact_lock:
            try:
                job = manager.completed_job(job_id)
                if not job or not 0 <= index < len(job['results']):
                    raise ValueError('Result is unavailable.')
                directory = job['results'][index]['directory']
                target = result_thumbnail(directory, manager.preview_root(directory))
                return Response(target.read_bytes(), media_type='image/jpeg',
                                headers={'Cache-Control': 'private, max-age=3600'})
            except (OSError, ValueError, KeyError, TypeError, av.error.FFmpegError) as exc:
                raise HTTPException(404, 'Preview unavailable.') from exc

    @app.delete('/api/jobs/{job_id}')
    def delete_job(job_id: str):
        with artifact_lock:
            try:
                return manager.delete_completed(job_id)
            except OSError as exc:
                raise HTTPException(400, f'Could not delete job: {exc}') from exc

    @app.get('/api/status')
    def status():
        report, logs = manager.snapshot()
        return {'report': report, 'logs': logs}

    @app.post('/api/cancel')
    def cancel():
        return {'message': manager.cancel()}

    def media_url(path):
        return '/media/previews/'+quote(str(Path(path).relative_to(previews)), safe='/')

    def scene_urls(scene):
        if scene.get('mesh'):
            scene['mesh']['url'] = media_url(scene['mesh']['url'])
        return scene

    @app.post('/api/layout')
    def layout(body: Settings):
        return {'scene': export_layout(body.options()),
                'note': 'Layout preview. Final framing is fitted to the SAM body predictions.'}

    def training_info(job_id, result):
        status = result['training']
        chunk = result['chunk']
        info = dict(status=status.get('status'), step=status.get('step', 0),
                    steps=status.get('steps', 0), model=None,
                    watch=f'/api/jobs/{job_id}/training?chunk={chunk}')
        output = Path(result['directory'])/'training'
        if not output.resolve().is_relative_to(manager.job_directory(job_id)):
            return info
        preview_step = status.get('preview_step', 0)
        completed = status.get('status') == 'completed'
        if not completed and (not isinstance(preview_step, int) or preview_step < 1):
            return info
        relative = 'scene.ftgs.ply' if completed else f'previews/step-{preview_step:09d}.ftgs.ply'
        try:
            path = within(output, relative)
            stat = path.stat()
            normalization = json.loads(within(output, 'normalization.json').read_text())
            transforms = {key:normalization[key] for key in ('world_to_training','training_to_world')}
        except (HTTPException, OSError, ValueError, KeyError, TypeError):
            return info
        model_url = f'/api/jobs/{job_id}/model?chunk={chunk}'
        if not completed:
            model_url += f'&preview={preview_step}'
        version = f'{stat.st_mtime_ns}-{stat.st_size}'
        info.update(model=model_url+'&v='+version, version=version, live=not completed,
                    preview_step=status.get('steps', 0) if completed else preview_step,
                    **transforms)
        return info

    def result_training(directory):
        directory = Path(directory).resolve()
        if directory.is_relative_to(root):
            parts = directory.relative_to(root).parts
            if parts and parts[0].startswith('job-'):
                for result in manager.training_results(parts[0]):
                    if result['directory'] == str(directory):
                        return training_info(parts[0], result)
        return None

    @app.post('/api/results/open')
    def open_result(body: ResultRequest):
        try:
            with artifact_lock:
                scene, videos, metadata, note = export_result(body.directory, manager.preview_root(body.directory))
        except (KeyError, OSError, TypeError) as exc:
            raise HTTPException(400, f'Cannot open result: {exc}') from exc
        directory = Path(body.directory).expanduser().resolve()
        return {'scene': scene_urls(scene), 'videos': [media_url(p) for p in videos],
                'metadata': metadata, 'note': note, 'directory': str(directory), 'training': result_training(directory)}

    @app.get('/api/jobs/{job_id}/training')
    def training(job_id: str, chunk: int = 0):
        result = next((r for r in manager.training_results(job_id) if r['chunk'] == chunk), None)
        if result is None:
            raise HTTPException(404, 'Training result not found.')
        return JSONResponse(training_info(job_id, result), headers={'Cache-Control':'no-store'})

    @app.get('/api/jobs/{job_id}/model')
    def trained_model(job_id: str, chunk: int = 0, preview: int = 0):
        with artifact_lock:
            result = next((r for r in manager.training_results(job_id) if r['chunk'] == chunk), None)
            if result is None:
                raise HTTPException(404, '4D scene not found.')
            status = result['training']
            if preview:
                if preview < 1 or preview > status.get('preview_step', 0):
                    raise HTTPException(404, 'Training preview not found.')
                relative = f'previews/step-{preview:09d}.ftgs.ply'
            else:
                if status.get('status') != 'completed':
                    raise HTTPException(404, '4D scene is not complete.')
                relative = 'scene.ftgs.ply'
            directory = manager.job_directory(job_id)
            path = Path(result['directory'])/'training'/relative
            return FileResponse(within(directory, str(path.relative_to(directory))),
                                media_type='application/octet-stream', filename='scene.ftgs.ply',
                                headers={'Cache-Control':'private, max-age=3600'})

    @app.get('/api/results')
    def results():
        paths = []
        for report in sorted(root.glob('job-*/results/streams_report.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True):
            try:
                for job in json.loads(report.read_text())['jobs']:
                    if has_generated_result(job) and Path(job['result_dir']).is_dir():
                        if job['result_dir'] not in paths:
                            paths.append(job['result_dir'])
            except (OSError, ValueError, KeyError):
                continue
        return paths

    @app.get('/media/{kind}/{relative:path}')
    def media(kind: str, relative: str):
        roots = {'uploads': uploads, 'previews': previews}
        if kind not in roots:
            raise HTTPException(404)
        return FileResponse(within(roots[kind], relative))

    @app.get('/api/jobs/preview')
    def preview(index: int = 0, version: str = ''):
        with artifact_lock:
            report, _ = manager.snapshot()
            if index < 0 or index >= len(report['jobs']):
                return {'pending': True}
            job = report['jobs'][index]
            if job['status'] != 'running':
                return {'pending': True}
            result = live_preview(job, manager.preview_root(job['request']), version)
            if job.get('result_dir'):
                result['training'] = result_training(job['result_dir'])
                result['directory'] = job['result_dir']
        if 'videos' in result:
            scene_urls(result['scene'])
            result['videos'] = [media_url(p) if p else None for p in result['videos']]
        return result

    @app.get('/ftgs/{module}')
    def ftgs_module(module: str):
        # Reuse the pinned parser and sorter in our existing Three.js viewer.
        if module not in ('ftgs.js', 'sort.js'):
            raise HTTPException(404)
        from fdanyone.reconstruction import REPO
        script = REPO/'player'/module
        if not script.is_file():
            raise HTTPException(404, 'Initialize the FreeTimeGsVanilla submodule for 4D playback.')
        return FileResponse(script, media_type='text/javascript', headers={'Cache-Control':'no-cache'})

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    app.mount('/three', StaticFiles(directory=assets), name='three')
    return app
