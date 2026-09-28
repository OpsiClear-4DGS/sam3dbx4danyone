# Output

With the default `--data_dir data`, each inference run writes:

```text
data/
└── fdanyone/<clip>/
    ├── preprocessing/
    │   ├── sam3d_predictions.npz
    │   ├── viewer_geometry.npz        # canonical SAM mesh and skeleton for the UI
    │   ├── report.json
    │   └── geometry_checks.json
    ├── metadata.json
    ├── cameras.json
    ├── skeletons/00.mp4 ... <N-1>.mp4
    └── videos/
        ├── sparse/*.mp4                 # RCP proposal views, when used
        └── dense/00.mp4 ... <N-1>.mp4  # final target views
```

`<clip>` is the input filename without its extension, and `N` is the number of target views.

## Files

- `videos/dense/` contains the generated target-view videos.
- `videos/sparse/` contains intermediate proposal views for larger camera layouts.
- `skeletons/` contains pose-conditioning videos aligned with the target views.
- `cameras.json` contains camera intrinsics and poses for every target view.
- `metadata.json` records the input, generation settings, model versions,
  timings, PyTorch allocated/reserved peaks, actual NVML process peaks, and
  per-rank metrics when target denoising uses multiple GPUs.
- `preprocessing/` stores native SAM 3D Body MHR parameters, keypoints, camera
  translations, actual backbone precision/backend, and source reprojection checks.
  These are per-frame predictions under a static-camera assumption.
- `preprocessing/viewer_geometry.npz` contains float32 `keypoints` (121 × 70 × 3),
  float32 `vertices` (121 × 18,439 × 3) and integer `faces` (36,874 × 3).
  Vertices use the same canonical world, meters and frame order as the camera rig.
  Faces are zero-based triangle indices. The fitted MHR surface is untextured;
  older files may contain only keypoints.

Source BiRefNet masks are computed once and shared by body inference and conditioning.
Source framing uses predicted in-frame landmarks; it does not claim independent
occlusion evidence.

All generated-view and skeleton videos contain 121 frames, use the selected output FPS, and contain no
audio. Generated dense/proposal views are 704×1280; skeleton videos retain the
canonical source raster so their projected coordinates remain exact.

The default `full` execution profile publishes `videos/dense/` with
RGB-lossless H.264 (`libx264rgb`, CRF 0), so decoding the saved target produces
the exact RGB bytes returned by the VAE. It uses x264's `veryfast` preset;
presets affect encode work and file size, not decoded RGB values. Proposal and
skeleton videos remain compact intermediates.

## Downstream reconstruction

Enable `--train_4dgs=True` to run the pinned FreeTimeGsVanilla submodule after
generation. It adds `training/` with straight RGBA training frames, COLMAP cameras,
mesh-derived initialization, checkpoints and `scene.tsog`; see
[training](training.md). Generated views and camera metadata remain unchanged
and can also be used with external reconstruction tools. Soft alpha supervises
object opacity, including empty space; RGB and prediction use matching random
backgrounds during optimization.
