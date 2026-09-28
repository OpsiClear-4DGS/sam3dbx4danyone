from __future__ import annotations

import unittest

import torch

from fdanyone.errors import FourDAnyoneError
from fdanyone.model.conditioning import (
    _PoseEncodingJob,
    _PoseFeatureBuilder,
    plan_pose_encoding,
)


class PoseEncodingPlanTests(unittest.TestCase):
    def test_low_vram_six_view_plan_uses_batch_one_and_separate_nulls(self) -> None:
        plan = plan_pose_encoding(
            num_features=6,
            group_size=6,
            packed_views=1,
            batch_limit=1,
            separate_nulls=True,
        )
        self.assertEqual(plan.batch_size, 1)
        self.assertTrue(plan.separate_nulls)
        self.assertEqual(plan.feature_groups, ((0,), (1,), (2,), (3,), (4,), (5,)))

    def test_rcp_plan_reserves_two_packed_views(self) -> None:
        plan = plan_pose_encoding(
            num_features=24,
            group_size=6,
            packed_views=2,
            batch_limit=4,
        )
        self.assertEqual(plan.batch_size, 4)
        self.assertEqual(len(plan.feature_groups), 12)
        self.assertTrue(all(len(group) == 2 for group in plan.feature_groups))

    def test_batch_limit_must_leave_room_for_a_feature(self) -> None:
        with self.assertRaisesRegex(FourDAnyoneError, "must exceed"):
            plan_pose_encoding(
                num_features=6,
                group_size=6,
                packed_views=2,
                batch_limit=2,
            )

    def test_separate_nulls_allow_batch_one_with_two_packed_features(self) -> None:
        plan = plan_pose_encoding(
            num_features=6,
            group_size=6,
            packed_views=2,
            batch_limit=1,
            separate_nulls=True,
        )
        self.assertEqual(plan.batch_size, 1)
        self.assertEqual(len(plan.feature_groups), 6)

    def test_real_only_job_does_not_copy_an_empty_null_slice(self) -> None:
        builder = _PoseFeatureBuilder(
            features=torch.zeros(1, 2),
            null_features=torch.full((1, 2), 7.0),
        )
        job = _PoseEncodingJob(
            feature_indices=(0,),
            skeletons=(),
            batch_size=1,
            packed_views=0,
            channels_last=False,
            null_only=False,
            builder=builder,
        )
        builder.store(job, torch.ones(1, 2))
        self.assertTrue(torch.equal(builder.features, torch.ones(1, 2)))
        self.assertTrue(torch.equal(builder.null_features, torch.full((1, 2), 7.0)))


if __name__ == "__main__":
    unittest.main()
