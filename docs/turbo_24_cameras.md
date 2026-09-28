# Experimental 24-camera Turbo

Turbo now accepts totals of 6, 8, 12, 16, 18, 24 or 36 cameras across up to three elevation rings. Six- and 24-camera single rings and an 18-camera/two-ring preset have completed generation runs; the other layouts have routing and UI tests only. Multi-ring and four-view-group Turbo quality has not been evaluated.

The UI presets use one ring for 6/8/12, two for 16/18/24, and three for 36. The pipeline prefers six-view groups, falling back to four when each ring is divisible by four but not six. For nine cameras per ring (18 total), six-view groups span ring boundaries. All four denoising steps cover every camera exactly once, using cyclic offsets `(0,2,4,0)`; RCP proposals are disabled. Existing single-ring routing is preserved.

This is a local extension of the six-camera Turbo path. No retraining or downstream 4D reconstruction evaluation has been performed.

```bash
.venv/bin/python inference.py \
  --video_path=data/source/pexels/5885633-hd_1080_1920_25fps.mp4 \
  --views_per_layer=24 \
  --data_dir=outputs/my-24-camera-run
```

Direct inference distributes denoising across available GPUs, up to four workers. Full-VAE decoding uses isolated GPU processes and restores camera ordering. Full-mode precision, compilation and memory defaults remain active.

The batch launcher assigns each input video to one GPU. With `--views_per_layer=24`, its four groups run sequentially on that GPU, so the latency differs from distributed single-video generation.

## Validation

The latest complete run took **446.50 seconds (7m26s)** with seven L40S GPUs available: four denoising workers and six decoding workers. Peak process VRAM was **21.21 GiB**, under the 23 GiB limit. All 24 videos and AV1 review copies decoded successfully: 121 frames at 704×1280 and 25 FPS. See [timings and quality limits](performance.md).

Generated validation videos, montages and contact sheets are not included in the source distribution.

Appearance varies between views; visual review does not establish downstream 4D consistency.
