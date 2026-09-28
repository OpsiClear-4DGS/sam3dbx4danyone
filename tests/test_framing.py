from __future__ import annotations

import unittest

import numpy as np

from fdanyone.geometry.framing import (
    NON_FULL_BODY_BOTTOM_CAP,
    analyze_input_framing,
)
from fdanyone.skeleton.keypoints import KEYPOINT_NAMES


def _geometry(frame_count: int) -> np.ndarray:
    points = np.zeros((frame_count, len(KEYPOINT_NAMES), 3), dtype=np.float32)
    points[..., 2] = 5.0
    vertical = {
        "nose": -1.0,
        "left-eye": -1.0,
        "right-eye": -1.0,
        "left-ear": -1.0,
        "right-ear": -1.0,
        "neck": -0.7,
        "left-shoulder": -0.5,
        "right-shoulder": -0.5,
        "left-hip": 0.0,
        "right-hip": 0.0,
        "left-knee": 0.4,
        "right-knee": 0.4,
        "left-ankle": 0.8,
        "right-ankle": 0.8,
        "left-big-toe-tip": 1.0,
        "left-small-toe-tip": 1.0,
        "left-heel": 1.0,
        "right-big-toe-tip": 1.0,
        "right-small-toe-tip": 1.0,
        "right-heel": 1.0,
    }
    lookup = {name: index for index, name in enumerate(KEYPOINT_NAMES)}
    for name, y in vertical.items():
        points[:, lookup[name], 1] = y
    return points


def _inputs(frame_count: int):
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsics = np.broadcast_to(intrinsics, (frame_count, 3, 3)).copy()
    landmarks_2d = np.zeros((frame_count, 17, 3), dtype=np.float32)
    landmarks_2d[..., :2] = 50.0
    landmarks_2d[..., 2] = 0.9
    masks = np.full((frame_count, 100, 100), 255, dtype=np.uint8)
    return intrinsics, landmarks_2d, masks


class InputFramingEvidenceTests(unittest.TestCase):
    def test_inferred_feet_cannot_claim_full_body_without_observation_evidence(
        self,
    ) -> None:
        frame_count = 8
        intrinsics, landmarks_2d, masks = _inputs(frame_count)

        profile = analyze_input_framing(
            _geometry(frame_count),
            KEYPOINT_NAMES,
            intrinsics,
            landmarks_2d,
            masks,
            np.zeros(frame_count, dtype=bool),
        )

        self.assertEqual(profile.label, "half_body")
        self.assertFalse(profile.full_body_observed)
        self.assertEqual(profile.full_body_evidence_ratio, 0.0)
        self.assertLessEqual(profile.visible_body_bottom, NON_FULL_BODY_BOTTOM_CAP)

    def test_observed_distal_leg_allows_full_body_classification(self) -> None:
        frame_count = 8
        intrinsics, landmarks_2d, masks = _inputs(frame_count)

        profile = analyze_input_framing(
            _geometry(frame_count),
            KEYPOINT_NAMES,
            intrinsics,
            landmarks_2d,
            masks,
            np.ones(frame_count, dtype=bool),
        )

        self.assertEqual(profile.label, "full_body")
        self.assertTrue(profile.full_body_observed)
        self.assertEqual(profile.full_body_evidence_ratio, 1.0)

    def test_full_body_evidence_must_share_the_frame_count(self) -> None:
        frame_count = 8
        intrinsics, landmarks_2d, masks = _inputs(frame_count)

        with self.assertRaisesRegex(ValueError, "full-body evidence"):
            analyze_input_framing(
                _geometry(frame_count),
                KEYPOINT_NAMES,
                intrinsics,
                landmarks_2d,
                masks,
                np.ones(frame_count - 1, dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
