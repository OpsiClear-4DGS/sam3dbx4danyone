# 4D scene training

[FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla) is pinned
as a Git submodule at `third_party/FreeTimeGsVanilla`. The pipeline generates
calibrated views, preserves BiRefNet soft masks and trains an animated Gaussian
scene initialized from the SAM body mesh.

## Setup

```bash
git submodule update --init --recursive
uv sync --locked --project third_party/FreeTimeGsVanilla
npm ci --ignore-scripts --prefix third_party/tsog
npm run build --prefix third_party/tsog
```

Follow the submodule's CUDA/compiler requirements for gsplat, fused-ssim and
torch-scatter. Training uses its own Python 3.12+ / NumPy 1 environment;
generation uses Python 3.11 / NumPy 2. The source checkout and initialized
submodule are required; the trainer is not bundled into Python wheels.

To reuse a compatible environment, set `FDANYONE_FREETIMEGS_PYTHON` to its
Python executable. Code always comes from this project's submodule. The
runtime check imports the actual trainer and verifies CUDA before generation.
LPIPS may download AlexNet weights into the trainer's Torch cache on first use.
Final packaging uses Node.js 22+ and the pinned original TSOG v4 encoder with
WebGPU/Vulkan. Its dependencies are fixed by its npm lockfile. FFmpeg with AAC
and `atempo` is required to preserve source audio.

## Run

In the job panel, run through **Splat 4DGS** to generate and train each chunk.
Select **Splat 4DGS** to train directly from an existing generation version, or
change its settings and rerun it. End at **Generate videos** to skip training. The model plays directly in
the existing Three.js viewer, sharing its timeline, camera controls and body
mesh. **Display → 4D scene** controls visibility; **Display → Download TSOG**
saves the completed container. **Display → Sound** enables its embedded audio.

```bash
# Generate and train one video, or process a queue.
.venv/bin/python inference.py --video_path=video.mp4 --train_4dgs=True
.venv/bin/python batch_inference.py \
  --video_paths='["first.mp4","second.mp4"]' \
  --output_dir=outputs/scenes --train_4dgs=True

# Train saved calibrated views without regenerating them.
.venv/bin/python train_4dgs.py --result_dir=data/fdanyone/video --gpu_id=0
```

The CLI queue generates and trains each chunk on the same GPU. The UI queues
all chunks of each selected stage before advancing to the next stage; a chunk
may use a different GPU in its next stage. Generation
weights are released first; the isolated trainer sees only that GPU's UUID.
Other GPUs process other chunks. A single scene does not span multiple GPUs,
and the generation VRAM cap does not apply to training.
The reference TSOG encoder selects its own WebGPU device, independently of
`CUDA_VISIBLE_DEVICES`. Final packaging is serialized across workers on this
host; generation and training retain their assigned CUDA devices.

The default is 30,000 steps with 32,768 mesh samples per keyframe. Generation
commands accept `--training_steps`; `train_4dgs.py` accepts `--steps` and
`--samples_per_keyframe`. Short runs are installation checks, not quality runs.

The default destination is `<result>/training`, which is never overwritten.
Failures preserve generated views, logs and partial training previews. Retry
with a new `--output_dir`; CLI custom output directories are outside gallery
management. UI training attempts instead live under
`job-*/stages/splat-*/<chunk>/training`, and every version remains available in
the job panel and its download URLs. Each chunk is an independent model; overlap does not stitch scenes.

## Training contract

1. Decode every frame of the original lossless `videos/dense` files. Browser
   playback copies are not training inputs. Reuse one BiRefNet instance across
   bounded batches of generated views, whose backgrounds can differ from the
   source video.
2. Save **straight RGBA PNGs**: original RGB plus soft mask alpha. Alpha is
   neither thresholded nor discarded. Dataset version 2 requires real 8-bit
   RGBA at every camera/time pair; legacy black-composited RGB must be rebuilt
   from the original videos. Black pixels never stand in for a mask.
3. Export the saved pinhole intrinsics and OpenCV camera poses to binary
   COLMAP models. Sample the animated SAM surface every five frames, including
   the last: 25 keyframes and 819,200 Gaussians for a 121-frame clip. Surface
   colors use visibility, mask and view-angle checks. Fixed triangle identity
   and barycentric coordinates provide velocities in meters per frame.
4. Convert time and velocity to normalized-time units together. For 121
   frames, `frame_end=121` is exclusive and time is divided by 120. Normalize
   scene center and scale while retaining canonical Y up. Train all cameras
   with packed rasterization, learned motion/temporal opacity and relocation.
5. Supervise opacity over the entire image with alpha L1 weight **0.1**. Apply
   RGB L1/SSIM/LPIPS on matching random backgrounds. For each step, sample one
   RGB color in [0, 1]: target = `RGB * alpha + background * (1 - alpha)`;
   prediction = premultiplied render + `background * (1 - rendered_alpha)`.
   Background errors remain in the loss. Resizing, undistortion and cropping
   transform RGB and alpha together.

This follows [Brush's transparent-alpha training approach](https://github.com/ArthurBrussee/brush/tree/6378a76add3b93501abb55c2dc08d71688537679),
using its 0.1 alpha-loss weight. Our background sampler uses uniform [0, 1]
colors instead of small noise around a base color. The PyTorch implementation
is original; no Brush source is bundled. The native trainer keeps RGB-only
training available for unrelated datasets; this pipeline selects transparency.

## Playback and files

The first complete live snapshot is exported near step 100, then at most once
every 30 seconds. Two snapshots are retained during training and removed when
the final model is published. Failed preview exports do not abort training;
failed final model export does fail the job. The viewer keeps the current model
until a replacement is ready, preserving camera, frame and visibility choices.
This is periodic preview, with download and rendering latency on the client.

```text
training/
  status.json, request.json, train.log
  dataset/
    manifest.json, init.npz
    images/camNNN/*.png                 # straight RGBA
    sparse/0/{cameras,images,points3D}.bin
  normalization.json                   # canonical ↔ training coordinates
  ckpts/ckpt_29999.pt
  scene.tsog                           # packaged continuous 4DGS, timing and audio
  previews/step-*.ftgs.ply             # temporary live snapshots
  cfg.yml, stats/, tb/, videos/
```

`status.json` records the trainer revision and source hashes. Use
`normalization.json` with external tools: the model uses normalized training
coordinates. The integrated viewer reuses the submodule's parser/sorter and
adapts its shaders to Three.js. It preserves source FPS and the complete time
range, but is not pixel-identical to CUDA. The trainer's optional trajectory
video may show a shorter interval than the complete animated container.

TSOG retains every Gaussian and SH3, with 16-bit motion and ten codebook
clustering iterations. It is quantized, not lossless. Live previews remain
temporary PLYs so packaging does not interrupt optimization; the final staging
PLY is removed after successful TSOG publication. The checkpoint retains the
original trained parameters. Failed packaging preserves both checkpoint and
staging PLY and does not mark the job complete.

When the source has audio, the first track is trimmed to the first and last
sampled frames, tempo-adjusted without changing pitch, and preserved as AAC
192 kbit/s in `<result>/audio.m4a`. Prepared chunk audio is reused without a
second encoding. TSOG embeds it as `audio/track.m4a`, together with the player's
optional `playback` and `audio` metadata. FPS and duration `(frames-1)/fps`
preserve the normalized animation endpoints. VFR input uses an average tempo
over each selected span, matching its first and last frames; within-span timing
can vary with the source's frame timestamps. Audio gaps become silence; silent
videos have no audio entry. Generated camera videos remain silent.

The viewer synchronizes embedded audio with playback, pause, seek and looping;
sound starts muted until enabled. Audio metadata is a FreeTimeGsVanilla player
extension, so other TSOG viewers may ignore it. See the [container metadata
contract](../third_party/FreeTimeGsVanilla/player/TSOG.md#audio-and-playback-metadata).

To convert an older completed result without retraining (the original PLY is
retained), run:

```bash
.venv/bin/python -m fdanyone.reconstruction.tsog /path/to/fdanyone/clip
```

Any existing `audio.m4a` is included. Legacy runs that discarded source audio
need that track recovered from the original upload before conversion.

## Measured quality and limits

A controlled 18-view test used 121 frames at 704×1280, 25 FPS, identical
initialization arrays, 819,200 Gaussians and 30,000 steps for both models.
CUDA evaluation sampled five frames from every training camera (90 pairs),
with no pruning or display filter applied to hide stray geometry.

| Measurement | RGB-only baseline | Transparent-alpha training |
| --- | ---: | ---: |
| Mean opacity outside foreground | 26.55% | 0.197% |
| Background pixels with opacity > 0.1 | 43.39% | 0.396% |
| Silhouette IoU | 0.5928 | 0.9503 |
| Gray-background PSNR / SSIM | 14.93 dB / 0.7364 | 28.55 dB / 0.9221 |
| Soft-mask-weighted foreground PSNR | 21.61 dB | 21.05 dB |

Background excludes a five-pixel band around target alpha > 2/255; silhouette
IoU uses alpha ≥ 0.5. Foreground PSNR weights premultiplied RGB squared error
by soft alpha. These are generated training-view measurements, not held-out
or real-world ground truth. Mean background opacity fell **99.26%** and large
black spikes disappeared in the checked views and moving-camera orbit.
Foreground detail is slightly softer (−0.56 dB). The masks and matching
backgrounds were tested together, not as separate ablations.

On one L40S, preparation through final export took **15 min 12 s**: 313.77 s
for RGBA preparation and 590.17 s for the optimization loop. This excludes
generation and the initial runtime check. The last live snapshot took 0.24 s
to export. Both comparison videos decoded as 121 H.264 frames at 25 FPS.
Full-model browser playback, frame scrubbing, visibility, failed-download
recovery, cancellation on result changes and 320–1440 px layouts passed.

TSOG packaging of that same 819,200-Gaussian model took **44.81 s**, reducing
219.55 MB to **20.62 MB** (10.65× smaller). Across the same 90 camera/frame pairs,
TSOG versus original renders measured **46.95 dB PSNR / 0.99756 SSIM**; target
reconstruction PSNR changed from 28.55 to **28.43 dB**. These measurements exclude
audio; the tested source video has no audio stream.

The initial surface is a fitted body. Hair, loose clothing and accessories
must be learned from generated views, whose consistency, motion errors and
mask errors limit quality. Alpha supervision improves transparency; it does
not guarantee perfect geometry or recover the scene background.

FreeTimeGsVanilla retains its [AGPL license](../third_party/FreeTimeGsVanilla/LICENSE).
Generation model terms remain in the [third-party notices](THIRD_PARTY_NOTICES.md).
