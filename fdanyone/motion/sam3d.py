"""Direct SAM 3D Body inference, localized and prompted by BiRefNet."""
from __future__ import annotations
import gc
import time
import cv2
import numpy as np
import torch
from fdanyone.motion.sam3d_runtime import _load_inference_runner, _sam3d_onnx_backbone, sam3d_imports
COCO_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]

def mask_boxes(masks):
    """Largest foreground component; intended only for single-person clips."""
    boxes = []
    for mask in masks:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask >= 128).astype(np.uint8), connectivity=8
        )
        if count < 2:
            raise ValueError("Empty BiRefNet mask; cannot locate the subject.")
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, _ = stats[index]
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, dtype=np.float32)


def summarize_motion(points, fps):
    # Image-space second differences also contain true acceleration.
    acceleration = np.linalg.norm(np.diff(points, n=2, axis=0), axis=-1)
    return {
        "second_difference_median_px_per_frame2": float(np.median(acceleration)),
        "second_difference_p95_px_per_frame2": float(np.percentile(acceleration, 95)),
        "fps": fps,
    }


def predict_body(clip, masks, assets, sam_root, device, *, backend="auto", precision="fp16", batch_size=8):
    """Infer native MHR70 geometry using the same masks used for conditioning."""
    timings = {}
    boxes = mask_boxes(masks)
    def tick():
        torch.cuda.synchronize(device)
        return time.perf_counter()
    with sam3d_imports(sam_root), torch.no_grad():
        start = tick()
        runner = _load_inference_runner(
            assets.sam3d_backbone_onnx,
            device_index=torch.cuda.current_device(),
            backend=backend, trt_precision=precision,
        )
        runner_backend, runner_precision = runner.backend_name, runner.precision
        with _sam3d_onnx_backbone(sam_root, runner):
            from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
            from sam_3d_body.data.utils.prepare_batch import prepare_batch
            from sam_3d_body.utils import recursive_to

            model, cfg = load_sam_3d_body(
                str(assets.sam3d_checkpoint), device=torch.device(device),
                mhr_path=str(assets.mhr_model),
            )
            estimator = SAM3DBodyEstimator(model, cfg)
            timings["sam3d_load_s"] = tick() - start
            outputs = {}
            start = tick()
            for offset in range(0, len(clip.frames), batch_size):
                batches = []
                for i in range(offset, min(offset + batch_size, len(clip.frames))):
                    batches.append(prepare_batch(
                        clip.rgb_frames[i], estimator.transform, boxes[i:i+1],
                        masks=(masks[i:i+1] >= 128).astype(np.uint8)[..., None],
                    ))
                batch = {}
                for key, value in batches[0].items():
                    batch[key] = (torch.cat([item[key] for item in batches])
                                  if isinstance(value, torch.Tensor)
                                  else [item[key] for item in batches])
                batch = recursive_to(batch, device)
                model._initialize_batch(batch)
                prediction = model.forward_step(batch, decoder_type="body")["mhr"]
                for key in ("pred_keypoints_2d", "pred_keypoints_3d", "pred_cam_t",
                            "body_pose", "hand", "shape", "scale", "global_rot",
                            "mhr_model_params", "focal_length"):
                    outputs.setdefault(key, []).append(prediction[key].float().cpu().numpy())
                print(f"SAM 3D Body: {min(offset + batch_size, len(clip.frames))}/{len(clip.frames)}", flush=True)
            timings["sam3d_inference_s"] = tick() - start
            mask_embed_type = cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None)
            del estimator, model
        del runner
    gc.collect()
    torch.cuda.empty_cache()
    result = {key: np.concatenate(value) for key, value in outputs.items()}

    points = result["pred_keypoints_2d"][:, COCO_IDS]
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Non-finite SAM 3D Body predictions")
    report = {
        "inference_backend": runner_backend, "inference_precision": runner_precision,
        "source": str(clip.source_path), "frames": len(clip.frames),
        "resolution": [clip.width, clip.height], "device": torch.cuda.get_device_name(),
        "timings": timings, "mode": "body", "mask_prompt": True,
        "mask_embedding_type": mask_embed_type,
        "box_policy": "largest BiRefNet foreground component, threshold 128",
        "sam3d": summarize_motion(points, float(clip.fps)),
        "limitations": ["Single-person heuristic; no identity tracker", "No temporal refinement",
                        "No ground truth; agreement is not reconstruction accuracy",
                        "Single timing run; load and first-use overhead included"],
    }
    return result, report
