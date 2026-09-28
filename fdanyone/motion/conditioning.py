"""Build multiview conditioning from direct native MHR predictions."""
import json

def body_geometry(predictions, mhr_model, device):
    """Evaluate saved MHR poses in the same canonical world used for conditioning."""
    import numpy as np
    import torch
    from fdanyone.skeleton.pipeline import _BodyGeometry

    p = predictions
    points_incam = p["pred_keypoints_3d"] + p["pred_cam_t"][:, None]
    # Re-evaluate the exact saved MHR identity/pose to obtain mesh and rig joints.
    # Official MHRHead converts centimeters to meters and flips Y/Z for OpenCV.
    model = torch.jit.load(str(mhr_model), map_location=device)
    vertices, joints = [], []
    with torch.inference_mode():
        for start in range(0, len(p["shape"]), 16):
            shape = torch.as_tensor(p["shape"][start:start+16], device=device)
            params = torch.as_tensor(p["mhr_model_params"][start:start+16], device=device)
            v, j = model(shape, params, torch.zeros((len(shape), 72), device=device))
            vertices.append(v.cpu().numpy() / 100)
            joints.append(j[..., :3].cpu().numpy() / 100)
    flip = np.array([1, -1, -1], dtype=np.float32)
    vertices_world = np.concatenate(vertices) + p["pred_cam_t"][:, None] * flip
    joints_world = np.concatenate(joints) + p["pred_cam_t"][:, None] * flip
    points_world = points_incam * flip
    # One static camera-to-world conversion for the whole sequence. Do not
    # remove per-frame translation or silently smooth away genuine motion.
    offset = (points_world[0, 9] + points_world[0, 10]) * .5
    offset[1] = vertices_world[..., 1].min()
    left = points_world[0, 9] - points_world[0, 10] + points_world[0, 5] - points_world[0, 6]
    left[1] = 0
    norm = np.linalg.norm(left)
    if norm < 1e-6:
        raise ValueError("Cannot establish subject orientation from degenerate body landmarks")
    left /= norm
    up = np.array([0, 1, 0], dtype=np.float32)
    rotation = np.stack([left, up, np.cross(left, up)])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ offset

    def canonical(value):
        return np.einsum("ij,tkj->tki", rotation, value - offset).astype(np.float32)

    geometry = _BodyGeometry(
        canonical(vertices_world), canonical(joints_world), canonical(points_world),
        points_incam, transform,
        {"model": "SAM-3D-Body MHR", "keypoints": "native MHR70",
         "world_policy": "static-camera; one sequence rigid transform",
         "shape_policy": "original per-frame predictions; no smoothing"},
    )
    faces = model.character_torch.mesh.faces.cpu().numpy().astype(np.int32)
    return geometry, faces


def save_viewer_geometry(path, geometry, faces):
    """Publish a complete animation atomically for live and saved viewers."""
    import numpy as np
    import uuid
    from pathlib import Path

    path = Path(path)
    temporary = path.with_name('.' + uuid.uuid4().hex + '.npz')
    try:
        np.savez_compressed(temporary, keypoints=geometry.keypoints_world,
                            vertices=geometry.vertices_world, faces=faces)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(clip, predictions, masks, assets, output, device, view_plan, *, canonical_video_path=None):
    import numpy as np
    from fdanyone.geometry.framing import analyze_input_framing
    from fdanyone.skeleton.keypoints import KEYPOINT_NAMES
    from fdanyone.skeleton.pipeline import build_geometry_conditioning
    from fdanyone.video import write_motion_video

    output.mkdir(parents=True, exist_ok=True)
    p = predictions
    if any(len(value) != len(clip.frames) or not np.isfinite(value).all() for value in p.values()):
        raise ValueError("SAM predictions must be finite and match the canonical timeline")
    points_incam = p["pred_keypoints_3d"] + p["pred_cam_t"][:, None]
    focal = p["focal_length"].reshape(-1)
    K = np.repeat(np.eye(3)[None], len(clip.frames), axis=0)
    K[:, 0, 0] = K[:, 1, 1] = focal
    K[:, 0, 2], K[:, 1, 2] = clip.width / 2, clip.height / 2
    projected = np.einsum("tij,tkj->tki", K, points_incam)
    projected = projected[..., :2] / projected[..., 2:3]
    projection_error = float(np.max(np.abs(projected - p["pred_keypoints_2d"])))
    if not np.isfinite(projection_error) or projection_error > .01:
        raise ValueError(f"Camera convention mismatch: {projection_error} px")

    geometry, faces = body_geometry(p, assets.mhr_model, device)
    save_viewer_geometry(output / "viewer_geometry.npz", geometry, faces)
    # Framing uses predicted in-frame landmarks, not independent detections.
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
    xy = p["pred_keypoints_2d"][:, ids]
    in_frame = ((xy[..., 0] >= 0) & (xy[..., 0] < clip.width)
                & (xy[..., 1] >= 0) & (xy[..., 1] < clip.height))
    observed = np.concatenate([xy, in_frame[..., None]], axis=-1)
    full_body = in_frame[:, [15, 16]].all(axis=1)
    framing = analyze_input_framing(points_incam, KEYPOINT_NAMES, K, observed, masks, full_body)
    conditioning = build_geometry_conditioning(
        geometry=geometry, masks=masks, input_framing=framing, clip=clip,
        canonical_video_path=canonical_video_path or write_motion_video(clip, output / "working.mp4"),
        output_dir=output / "conditioning", view_plan=view_plan,
        motion_world="sam3d_static_camera_y_up", body_model="mhr-sam3d",
        observation_policy="predicted_in_frame_landmarks_not_independent_visibility",
        front_direction=np.array([0., 0., -1.]),
    )
    (output / "geometry_checks.json").write_text(json.dumps({
        "max_source_reprojection_error_px": projection_error,
        "rotation_determinant": float(np.linalg.det(geometry.motion_world_to_canonical_world[:3, :3])),
        "ground_min_y": float(geometry.vertices_world[..., 1].min()),
        "source_framing": framing.to_dict(),
        "limitations": ["Static source camera assumed", "Per-frame shape and depth may fluctuate",
                        "Occlusion confidence is not independently measured"],
    }, indent=2) + "\n")
    return conditioning
