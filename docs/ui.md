# Interactive UI

The UI uses a Three.js scene with plain HTML/JavaScript controls and a FastAPI backend. It supports SAM 3D Body + BiRefNet, the full Wan VAE and the existing GPU queue, in the main NumPy 2 environment. ColmapView informed the scene layout and controls.

```bash
uv sync --extra trt --extra gui
.venv/bin/python app.py --server_port=8080
```

Open `http://127.0.0.1:8080`. To listen on the server network interfaces:

```bash
.venv/bin/python app.py --server_name=0.0.0.0 --server_port=8080
```

Use `http://YOUR_SERVER_IP:8080` on a network that permits this port. The UI is intended for a trusted local network and has no user authentication. For remote access, forward the local port over SSH:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-host
```

## Job workflow

**Add video** immediately creates a persistent job. The task panel has four steps:

| Step | Settings | Saved output |
| --- | --- | --- |
| Trim | Range and speed, 1×–4× | Lossless 121-frame chunks and matching audio |
| Extract pose | FP16 or FP32 | SAM body parameters, mesh and BiRefNet masks |
| Generate videos | Cameras, Turbo/Base and seed | Calibrated multiview videos |
| Splat 4DGS | Training steps and points per keyframe | Training snapshots and final TSOG |

Select any step to inspect or configure it. **Run through** chooses the last step
to execute: select the same step to run it alone, or a later step to continue the
pipeline. For a fresh upload, configure the steps, select **Trim**, and run through
**Splat 4DGS**. Select **Generate videos** as the end to skip training. The next
step is blocked until its required input is ready; missing or outdated inputs are
explained beside the run control.

Settings save automatically. Every execution creates a new output version with
its exact settings and upstream version IDs. Changing cameras reuses the selected
pose and masks; changing training settings reuses the selected videos. The
**Output version** menu shows the distinguishing settings. Selecting an older
version restores its matching upstream versions, without deleting any newer
outputs. Downstream versions that use different inputs are marked **earlier
inputs**. Editing settings alone does not change existing outputs; run the step
to apply them. Stopped or failed attempts retain logs and leave completed versions
available. Select an earlier completed version or rerun the failed step.

Click a gallery result while idle to reopen its task panel, settings and outputs.
Existing results from the previous UI are adopted in place. Old pose results
without saved masks recover the masks with BiRefNet on their next generation;
SAM inference is reused. Missing original uploads or working clips can prevent
rerunning early steps, while saved generated videos can still be used for training.

During a task, settings and version selection are locked. You can inspect steps,
browse other results, and return with **Active task**. Browsing another result
leaves the running job's controls intact. Closing a browser does not cancel work.
The page restores the active job, or the last job opened in that browser.

## Trim and cameras

1. The editor opens in the main workspace with a large preview and one thumbnail timeline. Drag the timeline edges to trim, click or drag its body to scrub, and click a numbered chunk marker to preview that chunk. The end handle gently snaps near whole-chunk boundaries relative to the selected start and speed; the highlighted handle, selection time and chunk count update together. Move past the small snap zone or hold Alt to trim freely. Keyboard trimming and exact start/end entry in the **…** menu bypass snapping. Snapping applies only while dragging the end handle. End time is exclusive. Use Space to play/pause, I/O to set trim boundaries at the playhead, and arrow keys on the focused playhead or trim handle for fine adjustment (Shift moves ten frames).
2. Choose **Speed**: 1× keeps every frame, 2× keeps every second frame, 3× keeps every third frame, and 4× keeps every fourth frame. The helper text shows the sampling choice. Kept frames play at the source frame rate. For variable-frame-rate sources, trimming uses decoded presentation timestamps and output uses the reported average frame rate; speed is approximate. The player previews the interval at the selected speed.
3. The app splits the kept frames into 121-frame chunks. Numbered markers and boundaries show them directly on the timeline. A partial last chunk overlaps the preceding chunk to include the selection's end, without padding or silently dropping it. At least 121 kept frames are required.
4. Use the step buttons to configure pose, cameras and training, or **Next** to advance. Click the source thumbnail to return to Trim. **Reset clip** in the **…** menu restores the full video at original speed. **Jobs** returns to the gallery and keeps the draft. Preview playback is muted.

The launcher decodes the source once into RGB-lossless working chunks, then submits all chunks to the existing queue. Each chunk retains exactly 121 selected source frames. The source frame rate is explicitly preserved through inference (including high-frame-rate inputs), so skipped frames accelerate motion rather than merely lowering output FPS. `job-*/stages/trim-*/clips/manifest.json` records source indices, trim times, overlap and frame rate. Source audio is trimmed and tempo-adjusted for each chunk, then embedded in its TSOG. Generated camera videos remain silent.

One persistent worker runs on each needed visible GPU. Use `CUDA_VISIBLE_DEVICES` before launch to restrict the pool. A shared task lock permits one pipeline at a time per cache directory, including across server instances. The queue preserves compiled model reuse and restarts failed workers. Chunks generate independently; results are not stitched into a continuous video, and consistency across chunk boundaries is not guaranteed. The API/CLI still accept batches of independent source videos without trim edits.

The sidebar shows the selected step's settings. Choosing a camera preset updates the layout in the scene immediately. Select an output version to return to its generated cameras. Presets:

| Cameras | Rings | Cameras per ring |
| --- | --- | --- |
| 6 / 8 / 12 | 1 | 6 / 8 / 12 |
| 16 | 2 | 8 |
| 18 | 2 | 9 |
| 24 | 2 | 12 |
| 36 | 3 | 12 |

One ring uses 15° elevation; two use 0°/30°; three use −15°/15°/45°. Each ring spans 360° with evenly spaced yaw angles, starting at 0°. The **…** menu beside Cameras holds the Turbo/Base selector and seed; camera layout preview updates automatically. The clip workflow preserves source frame rate and seed defaults to 42; custom poses and advanced overrides remain available through the API/CLI. The API's `views` field remains cameras **per ring**, with `pitches` specifying the rings; the UI converts the total accordingly.

Turbo is the default with full-mode optimizations. Added camera counts, four-view groups and multiple rings are experimental. Six- and 24-camera single rings and an 18-camera/two-ring preset have completed generation runs; other layouts have routing/UI tests only. Multi-ring reconstruction quality has not been evaluated. The 18-camera preset uses six-view groups crossing the nine-camera ring boundary. Base remains selectable. Layout preview shows nominal cameras; the final rig is fitted to body predictions.

Progress appears once in the sidebar. Single-chunk runs show a stage label and an activity bar, or training steps during optimization. Chunk counts appear only for multiple chunks. Expand the progress summary for logs and the active chunk selector. **View live** resumes following after manual browsing. **Stop** terminates the queue and GPU workers while preserving completed versions. The task panel stays open after completion.

Jobs and previews live in `outputs/ui/` by default; use `--cache_dir=/path/to/shared/results` for shared storage. A job's `job.json` records its source, draft configurations, selected versions, immutable attempt history and active run range. Each attempt has its own `stages/<stage>-<id>/` directory. In-progress attempts abandoned by a process or machine restart are marked interrupted when reopened. A graceful server shutdown stops its own queue; a second viewer does not stop another server's queue.

## Gallery and playback

**Jobs** shows drafts and saved scenes, newest first. Cards with outputs show three generated angles; multi-chunk jobs have a chunk selector. The gallery is hidden while working on a job and refreshes when you return. Search appears for six or more jobs. A card's trash control deletes the whole job, including every version, chunk, training artifact and owned preview. The original upload and model weights are kept. Active jobs cannot be deleted. Deleting an older job does not affect another running task.

External result folders can be viewed using `--output_dir=/path/to/result`; only owned jobs have task controls and gallery deletion.

Gaussian playback shares the Three.js scene, cameras and timeline. The first training snapshot appears near step 100, with updates about every 30 seconds, followed by the final model. **Display → Download TSOG** saves the completed container, even before it has loaded into WebGL. **Display → Sound** enables embedded audio and follows play, pause, seek and looping. Older PLY outputs retain their download until converted.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/uploads` | Save video and create a job; returns `job_id` |
| `GET /api/jobs` | Gallery, including drafts |
| `GET /api/jobs/{id}` | Configurations, versions, dependencies and active task |
| `PATCH /api/jobs/{id}` | Save `{configs: {stage: settings}}` |
| `POST /api/jobs/{id}/run` | Execute `{start, through, configs}` using `trim`, `pose`, `generate`, `splat` |
| `POST /api/jobs/{id}/select` | Select `{run_id}` and its upstream versions |
| `GET /api/jobs/{id}/stage?stage=pose&chunk=0&run=…` | View a particular stage attempt |
| `GET /api/jobs/{id}/training?chunk=0&run=…` | Training progress and immutable model URL |
| `DELETE /api/jobs/{id}` | Delete an inactive owned job |

Thumbnail and playback caches are grouped under `previews/job-<id>/`. Model URLs pin their training run, so changing versions cannot change an in-flight download. The legacy batch submission endpoint and CLI remain compatible.

The Three.js scene shows calibrated cameras with generated video planes and an adjacent selected-camera pane. Both use the same browser video elements. New results include the animated SAM 3D Body MHR mesh (18,439 vertices and 36,874 triangles), in the same canonical coordinates as the cameras. This is the fitted, untextured body surface; clothing, hair and scene surfaces still require reconstruction. The mesh appears only in the 3D scene. Older joint-only results retain their skeleton, and results without saved geometry show cameras and videos.

- Drag to orbit, right-drag to pan, scroll to zoom. Click a camera or use the camera selector.
- Play/Pause and the frame slider control the Gaussian model, body and every camera together. Scrubbing pauses and seeks all videos to the selected frame. During playback, the first available video is the clock and drift above 1.5 frames is corrected; browser playback is not a guaranteed frame-locked scientific video wall.
- The **Display** menu groups 4D scene, Grid, Cameras, Body and Videos visibility controls with Reset view and Fullscreen. **Body** toggles the mesh, or the skeleton when mesh data is unavailable. The body is initially hidden when the Gaussian scene first appears; it can be turned back on in the same coordinate system. Controls that do not apply are hidden or disabled. Reset view restores the orbit view; Fullscreen expands the scene. With the canvas focused, Space toggles playback and R resets the view.
- While following a running job, the viewer loads fitted cameras and the SAM mesh once preprocessing is complete, then adds each fully muxed generated video as it becomes available. During 4DGS optimization, the first model preview is published near step 100 and subsequent snapshots about every 30 seconds. Camera pose, paused frame, selected camera and visibility choices survive updates and the final handoff. Expand the progress summary to choose which chunk to follow. These are training snapshots, not intermediate diffusion frames or a continuous stream of every optimization step. Manual layout/result selection disables following so it does not overwrite your chosen scene.

Mesh vertices and topology are retained from the existing preprocessing pass, with no additional SAM inference. The browser fetches a cached binary animation, reuses GPU buffers on frame changes and releases them when switching results. Generated videos remain unchanged. An older result with saved SAM pose parameters can recover the surface through the MHR body model; the viewer falls back to its skeleton when no mesh has been saved.


Lossless RGB generation outputs remain unchanged. The UI creates H.264 YUV420 playback copies for browser compatibility, separately from originals. Opening a result for the first time requires transcoding; subsequent opens reuse a cache keyed by file identity. Preview compression is for display only.

Source inputs still require one person, at least 121 usable frames and a mostly stationary camera. No GVHMR, SMPL-X downloads or reconstruction service is used.

Three.js 0.186.1 downloads once on the server, verifies pinned SHA-512 integrity and stays in the external UI cache. Browser modules and videos are served from the same server, without a CDN. A first launch needs server-side internet access; later launches reuse cached assets. Rerun and Gradio are not dependencies.

UI source provenance is in [UPSTREAM.md](../fdanyone/space/UPSTREAM.md); dependency terms are included in the [license inventory](dependency_licenses.json).

## Validation

Run `.venv/bin/python -m unittest discover -s tests` for the CPU and mocked-inference suite. Node.js enables additional checks comparing browser timeline previews and snapped boundaries with the server planner across frame rates and playback speeds.

The tests cover frame-exact trimming and lossless chunk encoding, GPU queue handoff, saved-result management, camera calibration, progressive previews, mesh serialization and protection of source/model files during deletion. Browser validation covers native mouse/touch trimming, snap bypass, keyboard editing, saved scenes, shared-clock playback, all menus, and layouts from 320 to 1440 pixels wide. Browser checks use disposable results and mocked generation where needed; they do not constitute a new GPU generation benchmark. Test media and screenshots are kept outside the source distribution.

An earlier six-camera UI pipeline run completed in 411.20 seconds on one L40S, peaking at 20.71 GiB, with TensorRT FP16 SAM, compiled Turbo and the full VAE. It produced 121-frame, 704 × 1280, 25 FPS videos. The body scene appeared after preprocessing, and completed camera videos appeared progressively. Frustum vertices reprojected within 0.001 pixel of the saved calibration. See [performance and validation limits](performance.md) for other measurements.

The versioned workflow was exercised on an L40S with a real clip: trim 1.88 s,
pose 31.42 s, six-view Turbo generation from saved pose 377.11 s. Starting directly
at Splat 4DGS then produced a TSOG in 133.42 s using 501 steps and 4,096 points per
keyframe. That shortened training run validates integration, not reconstruction
quality. The default remains 30,000 steps and 32,768 points per keyframe. Browser
checks also cover draft restoration, independent trim execution, version lineage,
pose-only preview, TSOG download and layout widths 320–1440 pixels.
