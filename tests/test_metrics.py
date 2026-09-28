from __future__ import annotations

import unittest

from fdanyone.errors import FourDAnyoneError
from fdanyone.model.metrics import GenerationMetrics


class GenerationMetricsTests(unittest.TestCase):
    def test_cuda_and_process_peaks_merge_independently(self) -> None:
        metrics = GenerationMetrics(0)
        metrics.merge_cuda_peak(
            "denoise",
            allocated_bytes=10,
            reserved_bytes=20,
            process_bytes=30,
        )
        metrics.merge_cuda_peak(
            "denoise",
            allocated_bytes=12,
            reserved_bytes=18,
            process_bytes=25,
        )
        metrics.merge_cuda_peak(
            "decode",
            allocated_bytes=8,
            reserved_bytes=14,
            process_bytes=28,
        )

        self.assertEqual(
            metrics.stage_peak_vram_bytes,
            {
                "denoise": {"allocated": 12, "reserved": 20, "process": 30},
                "decode": {"allocated": 8, "reserved": 14, "process": 28},
            },
        )
        self.assertEqual(metrics.peak_vram_allocated_bytes, 12)
        self.assertEqual(metrics.peak_vram_reserved_bytes, 20)
        self.assertEqual(metrics.peak_vram_process_bytes, 30)

    def test_negative_process_measurement_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            GenerationMetrics(0).merge_cuda_peak(
                "invalid",
                allocated_bytes=0,
                reserved_bytes=0,
                process_bytes=-1,
            )

    def test_merged_worker_peak_is_subject_to_the_limit(self) -> None:
        metrics = GenerationMetrics(0, process_vram_limit_bytes=29)
        metrics.merge_cuda_peak(
            "distributed",
            allocated_bytes=10,
            reserved_bytes=20,
            process_bytes=30,
        )

        with self.assertRaisesRegex(FourDAnyoneError, "execution-profile limit"):
            metrics.enforce_process_limit("distributed")


if __name__ == "__main__":
    unittest.main()
