# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Isolated SAM 3D Body + BiRefNet preprocessing worker."""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from fdanyone.assets import resolve_sam3d_assets, resolve_foreground_model
from fdanyone.device import select_cuda_device
from fdanyone.errors import FourDAnyoneError
from fdanyone.model.vram import ProcessVramMonitor
from fdanyone.video import load_canonical_working_clip


def main(request_path: str) -> None:
    import numpy as np
    import torch
    from fdanyone.foreground import predict_foreground_masks
    from fdanyone.motion.sam3d import predict_body
    from fdanyone.motion.conditioning import prepare
    from fdanyone.views import ViewPlan
    logging.basicConfig(level=logging.INFO)
    request = json.loads(Path(request_path).read_text())
    device, index = select_cuda_device(request["device"])
    torch.cuda.set_device(index)
    torch.set_num_threads(8)
    monitor = ProcessVramMonitor(index).start()
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        clip = load_canonical_working_clip(request["working_video"], request["clip_metadata"])
        mask_started = time.monotonic()
        masks = predict_foreground_masks(clip.rgb_frames, resolve_foreground_model(request["model_dir"]), device)
        mask_elapsed = time.monotonic() - mask_started
        assets = resolve_sam3d_assets(request["model_dir"])
        predictions, report = predict_body(
            clip, masks, assets, Path(request["source"]), device,
            backend=request["backend"], precision=request["precision"],
        )
        report["timings"]["birefnet_load_and_inference_s"] = mask_elapsed
        report["foreground_preprocessing"] = "gpu_bilinear_antialias_fp16"
        report["timeline"] = {"fps_num": clip.fps_num, "fps_den": clip.fps_den, "start_time": str(clip.start_time)}
        np.savez_compressed(output / "sam3d_predictions.npz", **predictions)
        prepare(clip, predictions, masks, assets, output, device, ViewPlan.from_dict(request["view_plan"]),
                canonical_video_path=Path(request["working_video"]))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        peak = monitor.stop()
    limit = request.get("process_vram_limit_bytes")
    if limit is not None and peak >= limit:
        raise FourDAnyoneError(f"Preprocessing exceeded process VRAM limit: {peak / 2**30:.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1])
