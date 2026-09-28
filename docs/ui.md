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

Upload one video, then follow **Edit clip → Cameras → Generate**:

1. The editor opens in the main workspace with a large preview and one thumbnail timeline. Drag the timeline edges to trim, click or drag its body to scrub, and click a numbered chunk marker to preview that chunk. The end handle gently snaps near whole-chunk boundaries relative to the selected start and speed; the highlighted handle, selection time and chunk count update together. Move past the small snap zone or hold Alt to trim freely. Keyboard trimming and exact start/end entry in the **…** menu bypass snapping. Snapping applies only while dragging the end handle. End time is exclusive. Use Space to play/pause, I/O to set trim boundaries at the playhead, and arrow keys on the focused playhead or trim handle for fine adjustment (Shift moves ten frames).
2. Choose **Speed**: 1× keeps every frame, 2× keeps every second frame, 3× keeps every third frame, and 4× keeps every fourth frame. The helper text shows the sampling choice. Kept frames play at the source frame rate. For variable-frame-rate sources, trimming uses decoded presentation timestamps and output uses the reported average frame rate; speed is approximate. The player previews the interval at the selected speed.
3. The app splits the kept frames into 121-frame chunks. Numbered markers and boundaries show them directly on the timeline. A partial last chunk overlaps the preceding chunk to include the selection's end, without padding or silently dropping it. At least 121 kept frames are required.
4. Click **Choose cameras**, choose the camera count and review the output total (chunks × cameras). **Generate N videos** submits all chunks. Click the source thumbnail/name in the sidebar to return to editing. **Back** above the editor restores the last confirmed trim, speed and chunk plan. **Reset clip** in the **…** menu restores the full video at original speed. Preview playback is muted.

The launcher decodes the source once into RGB-lossless working chunks, then submits all chunks to the existing queue. Each chunk retains exactly 121 selected source frames. The source frame rate is explicitly preserved through inference (including high-frame-rate inputs), so skipped frames accelerate motion rather than merely lowering output FPS. `job-*/clips/manifest.json` records source indices, trim times, overlap and frame rate. Audio is not included in generated videos.

One persistent worker runs on each needed visible GPU. Use `CUDA_VISIBLE_DEVICES` before launch to restrict the pool. Only one batch runs through a UI instance at a time. The queue preserves compiled model reuse and restarts failed workers. Chunks generate independently; results are not stitched into a continuous video, and consistency across chunk boundaries is not guaranteed. The API/CLI still accept batches of independent source videos without trim edits.

The main sidebar has one source card with Edit/Replace actions, a compact edit summary, Cameras and Generate. The upload drop area is replaced by that card after an upload; output totals appear once on the Generate button. The editor has one primary action; chunk previews share the timeline and secondary controls use an overflow menu. The camera layout fills the workspace without an empty video pane or inactive timeline. Choose the total camera count; the scene preview updates immediately, with elevation rings and yaw spacing chosen automatically. Selecting a preset exits live/result playback and shows the planned cameras; select a saved result to return to playback. Presets:

| Cameras | Rings | Cameras per ring |
| --- | --- | --- |
| 6 / 8 / 12 | 1 | 6 / 8 / 12 |
| 16 | 2 | 8 |
| 18 | 2 | 9 |
| 24 | 2 | 12 |
| 36 | 3 | 12 |

One ring uses 15° elevation; two use 0°/30°; three use −15°/15°/45°. Each ring spans 360° with evenly spaced yaw angles, starting at 0°. The **…** menu beside Cameras holds the Turbo/Base selector; camera layout preview updates automatically. The clip workflow preserves source frame rate and seed defaults to 42; custom poses and advanced overrides remain available through the API/CLI. The API's `views` field remains cameras **per ring**, with `pitches` specifying the rings; the UI converts the total accordingly.

Turbo is the default with full-mode optimizations. Added camera counts, four-view groups and multiple rings are experimental. Six- and 24-camera single rings and an 18-camera/two-ring preset have completed generation runs; other layouts have routing/UI tests only. Multi-ring reconstruction quality has not been evaluated. The 18-camera preset uses six-view groups crossing the nine-camera ring boundary. Base remains selectable. Layout preview shows nominal cameras; the final rig is fitted to body predictions.

Preparation and chunk completion progress appear once in the sidebar while generation runs. Single-clip jobs show a stage label and an activity bar; chunk counts appear only for batches with multiple chunks. Expand the progress summary to choose a chunk and inspect logs. **View live** appears when browsing results has interrupted live following. **Stop** stops the launcher and its GPU workers while preserving completed results. Finished jobs move directly into the gallery and the progress panel disappears. Stopping the server also stops its active batch; closing the browser does not cancel generation. Jobs and previews live in `outputs/ui/` by default; set `--cache_dir=/path/to/shared/results` to use shared storage.

## View results

Completed jobs open and play automatically while following the active generation. On a fresh page, the latest saved result opens automatically. **Results** is a thumbnail gallery in the left sidebar, newest first. Click a card to open its scene; the current result has a highlighted border. Each preview shows three generated camera angles, with the source name, date and view count. Multi-chunk results have a chunk selector directly on the card. The selected scene’s name, camera count and duration appear above the viewer. On small screens the cards form a horizontal scrollable row; opening one scrolls to the scene. The gallery refreshes automatically when jobs change and every 15 seconds, preserving selection and avoiding redraws when nothing changed. Existing external result directories containing `metadata.json` and `cameras.json` can still be opened at launch:

```bash
.venv/bin/python app.py --output_dir=/path/to/result
```

Search appears above the gallery when there are six or more saved jobs. New uploads retain their original filenames; older uploads without name records appear as **Video**, distinguished by date and view count. The gallery survives server restarts. It is hidden while uploading, editing, choosing cameras and following a new generation. **← Results** in the scene header returns to saved scenes without discarding the prepared clip or stopping generation. The gallery returns automatically when that generation finishes; Back from the editor restores the screen you came from.

The trash icon on each card opens a confirmation showing its name, chunk count and saved file size. **Delete** permanently removes that job's results, prepared chunks, logs and owned previews, including every chunk. The original upload and model weights are retained. Running and unfinished jobs cannot be deleted through this control, and paths outside the UI's own job directories are never deletion targets. Deleting the displayed job opens the next result; deleting the last one clears playback. A different active generation job continues normally. Manually opened external result folders are viewable but are not managed/deleted by the gallery.

The API exposes `GET /api/jobs` for completed batches, `GET /api/jobs/{job_id}/thumbnail?index=0` for a chunk's JPEG preview and `DELETE /api/jobs/{job_id}` for a completed batch. Thumbnails decode one frame from at most three views on CPU and cache a 480 × 288 JPEG; browsing the gallery does not export playback videos or load models. Playback copies and thumbnails are grouped under `previews/job-<id>/`; matching older playback caches are adopted when results open. Upload names are stored beside their uploads and in each job's `ui.json`, separately from the generation request.

The Three.js scene shows calibrated cameras with generated video planes and an adjacent selected-camera pane. Both use the same browser video elements. New results include the animated SAM 3D Body MHR mesh (18,439 vertices and 36,874 triangles), in the same canonical coordinates as the cameras. This is the fitted, untextured body surface; clothing, hair and scene surfaces still require reconstruction. The mesh appears only in the 3D scene. Older joint-only results retain their skeleton, and results without saved geometry show cameras and videos.

- Drag to orbit, right-drag to pan, scroll to zoom. Click a camera or use the camera selector.
- Play/Pause and the frame slider control the body and every camera together. Scrubbing pauses and seeks all videos to the selected frame. During playback, the first available video is the clock and drift above 1.5 frames is corrected; browser playback is not a guaranteed frame-locked scientific video wall.
- The **Display** menu groups Grid, Cameras, Body and Videos visibility controls with Reset view and Fullscreen. **Body** toggles the mesh, or the skeleton when mesh data is unavailable. Controls that do not apply to a camera-layout preview are hidden. Reset view restores the orbit view; Fullscreen expands the scene. With the canvas focused, Space toggles playback and R resets the view.
- While following a running job, the viewer loads fitted cameras and the SAM mesh once preprocessing is complete, then adds each fully muxed generated video as it becomes available. Expand the progress summary to choose which chunk to follow. This previews completed stages, not intermediate diffusion steps. Manual layout/result selection disables following so it does not overwrite your chosen scene.

Mesh vertices and topology are retained from the existing preprocessing pass, with no additional SAM inference. The browser fetches a cached binary animation, reuses GPU buffers on frame changes and releases them when switching results. Generated videos remain unchanged. An older result with saved SAM pose parameters can recover the surface through the MHR body model; the viewer falls back to its skeleton when no mesh has been saved.


Lossless RGB generation outputs remain unchanged. The UI creates H.264 YUV420 playback copies for browser compatibility, separately from originals. Opening a result for the first time requires transcoding; subsequent opens reuse a cache keyed by file identity. Preview compression is for display only.

Source inputs still require one person, at least 121 usable frames and a mostly stationary camera. No GVHMR, SMPL-X downloads or reconstruction service is used.

Three.js 0.186.1 downloads once on the server, verifies pinned SHA-512 integrity and stays in the external UI cache. Browser modules and videos are served from the same server, without a CDN. A first launch needs server-side internet access; later launches reuse cached assets. Rerun and Gradio are not dependencies.

UI source provenance is in [UPSTREAM.md](../fdanyone/space/UPSTREAM.md); dependency terms are included in the [license inventory](dependency_licenses.json).

## Validation

Run `.venv/bin/python -m unittest discover -s tests` for the CPU and mocked-inference suite. Node.js enables additional checks comparing browser timeline previews and snapped boundaries with the server planner across frame rates and playback speeds.

The tests cover frame-exact trimming and lossless chunk encoding, GPU queue handoff, saved-result management, camera calibration, progressive previews, mesh serialization and protection of source/model files during deletion. Browser validation covers native mouse/touch trimming, snap bypass, keyboard editing, saved scenes, shared-clock playback, all menus, and layouts from 320 to 1440 pixels wide. Browser checks use disposable results and mocked generation where needed; they do not constitute a new GPU generation benchmark. Test media and screenshots are kept outside the source distribution.

An observed six-camera UI pipeline run completed in 411.20 seconds on one L40S, peaking at 20.71 GiB, with TensorRT FP16 SAM, compiled Turbo and the full VAE. It produced 121-frame, 704 × 1280, 25 FPS videos. The body scene appeared after preprocessing, and completed camera videos appeared progressively. Frustum vertices reprojected within 0.001 pixel of the saved calibration. See [performance and validation limits](performance.md) for other measurements.
