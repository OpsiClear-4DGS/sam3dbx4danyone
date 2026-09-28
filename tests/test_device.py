from __future__ import annotations

import unittest
from unittest.mock import patch

from fdanyone.device import CUDA_ALLOCATOR_CONF, configure_inference_cuda_allocator
from fdanyone.device import default_cpu_threads, configure_inference_cpu_environment


class CpuThreadBudgetTests(unittest.TestCase):
    def test_worker_budget_scales_with_available_cpus(self):
        with patch('os.sched_getaffinity', return_value=set(range(128))):
            self.assertEqual(default_cpu_threads(8), 8)
        with patch('os.sched_getaffinity', return_value=set(range(16))):
            self.assertEqual(default_cpu_threads(8), 2)
            self.assertEqual(default_cpu_threads(32), 1)

    def test_explicit_thread_limits_are_preserved(self):
        import os
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '3'}):
            configure_inference_cpu_environment()
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '3')
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '2'}, clear=True):
            configure_inference_cpu_environment()
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '2')


class CudaAllocatorConfigurationTests(unittest.TestCase):
    def test_low_vram_options_are_added(self) -> None:
        environment: dict[str, str] = {}

        configure_inference_cuda_allocator(
            environment,
            max_split_size_mb=4096,
            expandable_segments=True,
            garbage_collection_threshold=0.70,
        )

        self.assertEqual(
            environment[CUDA_ALLOCATOR_CONF],
            "max_split_size_mb:4096,expandable_segments:True,garbage_collection_threshold:0.7",
        )

    def test_caller_options_are_preserved_without_duplicates(self) -> None:
        environment = {
            CUDA_ALLOCATOR_CONF: (
                "backend:native,max_split_size_mb:512,"
                "expandable_segments:False,garbage_collection_threshold:0.8"
            )
        }

        configure_inference_cuda_allocator(
            environment,
            max_split_size_mb=4096,
            expandable_segments=True,
            garbage_collection_threshold=0.70,
        )

        self.assertEqual(
            environment[CUDA_ALLOCATOR_CONF],
            "backend:native,max_split_size_mb:512,"
            "expandable_segments:False,garbage_collection_threshold:0.8",
        )

    def test_garbage_collection_threshold_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            configure_inference_cuda_allocator({}, garbage_collection_threshold=1.0)


if __name__ == "__main__":
    unittest.main()
