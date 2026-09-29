# Performance and execution defaults

The `full` profile uses SAM 3D Body + BiRefNet, four-step Turbo, BF16 denoising and source encoding, FP32 denoising state, and full Wan VAE decoding in FP16 with rounded RGB output. The process VRAM limit is 23 GiB. `quality` remains a compatibility alias.

## Enabled optimizations

- SageAttention 2.2.0 on the validated L40S installation. The selected backend is logged and recorded in metadata. See [wheel compatibility](../wheels/README.md).
- Bounded attention compilation with `max-autotune-no-cudagraphs`. Outer one-view loops remain eager to limit temporary memory. `--compile_dit=False` selects the eager reference. Compilation is disabled for Base generation or when a selected GPU has less than 23 GiB free at startup.
- Up to eight CPU threads per worker, divided across available CPU affinity. Explicit OMP/MKL thread limits take precedence.
- One persistent batch worker per GPU, caching the denoiser in CPU memory between jobs and reusing compiled attention. Failed jobs restart their worker. See [GPU queues](multi_stream.md).
- GPU BiRefNet preprocessing and validated SAM 3D Body TensorRT FP16 engines when available; CUDA ONNX Runtime is the automatic fallback.
- Full-VAE FP16 decode tiles of size `(52,44)` and stride `(28,44)`, in isolated GPU subprocesses. Original checkpoint weights are cast directly to FP16; source encoding retains BF16 and its original tiling. Camera order is restored after distributed decoding.
- Decoded RGB is rounded to the nearest integer and saved losslessly. Non-finite pixels fail before export. Reference JPEGs and target videos share this conversion.

There are no alternate decoder precision or untiled execution modes. Tiny VAE, FP8 and whole-block compilation are absent.

## Decoder validation

The integrated production path decoded and published six 121-frame, 704×1280 videos on six L40S GPUs in **40.28 s**, including shard serialization, isolated worker startup, model loading and lossless export. Peak process VRAM was **15.42 GiB per GPU**. The earlier controlled FP16 comparison took 40.15 s versus 38.99 s for BF16, excluding shard serialization. These are single measured runs; source encoding, denoising and reconstruction are excluded.

In a fixed-input comparison, uploaded-source round-trip PSNR increased from 36.818 to 36.949 dB. Mean foreground PSNR on two generated-view round trips increased from 35.882 to 36.154 dB. Average SSIM and LPIPS also improved; the visual gain is subtle. No downstream 4DGS improvement has been measured. Production publication passed the 23 GiB guard and matched the selected experiment pixel-for-pixel across all six cameras and 726 frames. Source latents were unchanged; reference JPEG publication and BF16 re-encoding also passed.

## Measured complete runs

These historical measurements use 121 frames per camera at 704×1280 and 25 FPS, with the previous BF16 full-VAE decoder and eight CPU threads per worker. They have not been remeasured end to end with FP16 decoding.

| Run | GPU allocation | Complete pipeline | Denoising stage | Decode/publication | Peak process VRAM |
|---|---|---:|---:|---:|---:|
| Six views | One L40S | 423.10 s | 126.44 s | 206.47 s | 20.71 GiB |
| Experimental 24 cameras | Seven L40S available; four denoise, six decode | 446.50 s | 132.50 s | 141.09 s | 21.21 GiB |

These are individual runs with concurrent system activity, not throughput guarantees. Full-VAE decoding remains the largest single-GPU stage. They do not support a two-minute complete-pipeline claim. Eight simultaneous full-pipeline jobs have not been benchmarked.

## Quality and validation limits

A captured-input comparison measured warm denoising at 113.03 seconds with bounded attention compilation versus 126.54 seconds eager. All 726 decoded frames were compared: agreement with eager output was 40.26–46.50 dB PSNR across six videos. Reviewed frames closely matched identity, pose and clothing, but were not bit-identical. PSNR against eager output does not measure ground-truth reconstruction quality.

Complete pipeline runs separately passed the 23 GiB process cap. The captured-input benchmark retained extra tensors and is not the source of the production memory claim. Two successive jobs also completed in one persistent worker under the cap. Reusing identical conditioning preserved exact final latents; independently rerunning SAM preprocessing can introduce small differences.

The 24-camera path is experimental and has no downstream reconstruction benchmark. See [camera layout details](turbo_24_cameras.md).

Benchmark artifacts and generated playback copies are retained separately and are not part of the source distribution.
