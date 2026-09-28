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
    from fdanyone.motion.conditioning import prepare, body_geometry, save_viewer_geometry
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
        import hashlib
        digest = hashlib.sha256()
        for frame in clip.rgb_frames:
            digest.update(frame)
        identity = dict(rgb_sha256=digest.hexdigest(), fps=str(clip.fps), width=clip.width, height=clip.height)
        saved = Path(request["saved_pose_dir"]) if request.get("saved_pose_dir") else None
        assets = resolve_sam3d_assets(request["model_dir"])
        if saved:
            report = json.loads((saved / "report.json").read_text())
            if report.get("clip_identity") and report["clip_identity"] != identity:
                raise ValueError("Saved pose does not match this clip. Extract pose again.")
            with np.load(saved / "sam3d_predictions.npz", allow_pickle=False) as archive:
                predictions = dict(archive)
        mask_started = time.monotonic()
        if saved and (saved / "foreground_masks.npz").is_file():
            with np.load(saved / "foreground_masks.npz", allow_pickle=False) as archive:
                masks = archive["masks"]
        else:
            masks = predict_foreground_masks(clip.rgb_frames, resolve_foreground_model(request["model_dir"]), device)
        mask_elapsed = time.monotonic() - mask_started
        if not saved:
            predictions, report = predict_body(
                clip, masks, assets, Path(request["source"]), device,
                backend=request["backend"], precision=request["precision"],
            )
            report["timings"]["birefnet_load_and_inference_s"] = mask_elapsed
            report["foreground_preprocessing"] = "gpu_bilinear_antialias_fp16"
            report["timeline"] = {"fps_num": clip.fps_num, "fps_den": clip.fps_den, "start_time": str(clip.start_time)}
        report["clip_identity"] = identity
        if saved:
            report["reused_pose"] = str(saved)
        np.savez_compressed(output / "sam3d_predictions.npz", **predictions)
        if request.get("pose_only"):
            np.savez_compressed(output / "foreground_masks.npz", masks=masks)
            geometry, faces = body_geometry(predictions, assets.mhr_model, device)
            save_viewer_geometry(output / "viewer_geometry.npz", geometry, faces)
        else:
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
