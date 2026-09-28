# Generate videos on all GPUs

`batch_inference.py` automatically runs **one video job per visible GPU**. With eight GPUs and eight or more input videos, eight jobs run concurrently. When a job finishes, that GPU takes the next queued video. With fewer videos, only the needed GPUs are used.

```bash
.venv/bin/python batch_inference.py \
  --video_paths='["data/source/pexels/5885633-hd_1080_1920_25fps.mp4","data/source/pexels/2785536-uhd_2160_3840_25fps.mp4"]' \
  --output_dir=outputs/my-videos
```

Add more paths to generate more videos concurrently. Each job uses its GPU for the complete SAM 3D Body + BiRefNet and video-generation pipeline. GPU assignment is automatic; every worker sees only its own GPU. The launcher respects `CUDA_VISIBLE_DEVICES` if the environment already restricts GPU access.

The default is six-view, four-step Turbo generation with SageAttention, GPU foreground preprocessing and faster full-VAE decoding; see [speed defaults](performance.md). Each input needs 121 usable frames. Standard generation settings from `inference.py` are supported. Use a new output directory for each batch. Numbered job directories contain results and `worker.log`; `streams_report.json` records progress, GPU assignments and failures. Failed jobs do not stop the remaining queue. Ctrl-C stops active jobs and preserves completed outputs.

Each GPU keeps one worker alive for the queue. It retains one DiT in CPU memory between jobs and reuses its compiled attention code; weights return to the GPU only for denoising. Checkpoint/mode changes invalidate that cache, and the Turbo adapter is fused once. Preprocessing and full-VAE decoding remain isolated subprocesses. A failed job restarts that GPU's worker before continuing. More GPUs allow more videos at once.

Workers default to up to eight CPU threads each, divided according to available CPU affinity and the number of active workers. Explicit OMP/MKL thread limits take precedence.

With `--train_4dgs=True`, each slot also prepares foreground views and trains
FreeTimeGS on its assigned GPU before taking the next input. The cached
generation model is released before training; the next input reloads it.
Training uses the pinned submodule's separate environment and does not inherit
the generation allocator cap. See [4DGS training](training.md).

## Validation

Scheduler tests cover GPU discovery, one-GPU isolation, concurrent jobs, queue reuse, failure handling and cancellation. Two real single-GPU jobs completed concurrently during validation. Eight simultaneous full-pipeline jobs have not been benchmarked. See [current performance measurements](performance.md).
