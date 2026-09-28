# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Stage-local wall-time and CUDA-memory measurements for generation."""

from __future__ import annotations

import time
from contextlib import contextmanager

from fdanyone.model.vram import ProcessVramMonitor


class GenerationMetrics:
    """Measure independent stages while preserving an end-to-end CUDA peak."""

    def __init__(self, device_index: int, *, process_vram_limit_bytes: int | None = None) -> None:
        if process_vram_limit_bytes is not None and process_vram_limit_bytes <= 0:
            raise ValueError("process_vram_limit_bytes must be positive when provided.")
        self.device_index = device_index
        self.process_vram_limit_bytes = process_vram_limit_bytes
        self.elapsed_seconds: dict[str, float] = {}
        self.stage_peak_vram_bytes: dict[str, dict[str, int]] = {}
        self._active_stage: str | None = None

    @contextmanager
    def stage(self, name: str):
        """Measure one non-overlapping stage."""

        import torch

        if self._active_stage is not None:
            raise RuntimeError(f"Cannot start stage {name!r} while {self._active_stage!r} is active.")
        if name in self.elapsed_seconds:
            raise RuntimeError(f"Generation stage {name!r} was measured more than once.")

        self._active_stage = name
        torch.cuda.synchronize(self.device_index)
        torch.cuda.reset_peak_memory_stats(self.device_index)
        process_monitor = ProcessVramMonitor(self.device_index).start()
        started = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device_index)
            process_peak_bytes = process_monitor.stop()
            self.elapsed_seconds[name] = time.monotonic() - started
            self.merge_cuda_peak(
                name,
                allocated_bytes=int(torch.cuda.max_memory_allocated(self.device_index)),
                reserved_bytes=int(torch.cuda.max_memory_reserved(self.device_index)),
                process_bytes=process_peak_bytes,
            )
            self._active_stage = None
            self.enforce_process_limit(name)

    def merge_cuda_peak(
        self,
        name: str,
        *,
        allocated_bytes: int,
        reserved_bytes: int,
        process_bytes: int | None = None,
    ) -> None:
        """Merge a child-process/device measurement into a named stage."""

        if allocated_bytes < 0 or reserved_bytes < 0 or (process_bytes is not None and process_bytes < 0):
            raise ValueError("CUDA memory measurements cannot be negative.")
        previous = self.stage_peak_vram_bytes.get(name, {})
        self.stage_peak_vram_bytes[name] = {
            "allocated": max(allocated_bytes, previous.get("allocated", 0)),
            "reserved": max(reserved_bytes, previous.get("reserved", 0)),
            "process": max(process_bytes or 0, previous.get("process", 0)),
        }

    def enforce_process_limit(self, name: str) -> None:
        """Apply the profile gate after local or worker peaks have been merged."""

        record = self.stage_peak_vram_bytes[name]
        process_peak_bytes = record["process"]
        if self.process_vram_limit_bytes is None or process_peak_bytes < self.process_vram_limit_bytes:
            return
        from fdanyone.errors import FourDAnyoneError

        raise FourDAnyoneError(
            f"Generation stage {name!r} reached {process_peak_bytes / 2**30:.2f} GiB process VRAM; "
            f"PyTorch allocated {record['allocated'] / 2**30:.2f} GiB and reserved "
            f"{record['reserved'] / 2**30:.2f} GiB. The execution-profile limit is "
            f"{self.process_vram_limit_bytes / 2**30:.2f} GiB."
        )

    @property
    def peak_vram_allocated_bytes(self) -> int:
        return max((record["allocated"] for record in self.stage_peak_vram_bytes.values()), default=0)

    @property
    def peak_vram_reserved_bytes(self) -> int:
        return max((record["reserved"] for record in self.stage_peak_vram_bytes.values()), default=0)

    @property
    def peak_vram_process_bytes(self) -> int:
        return max((record["process"] for record in self.stage_peak_vram_bytes.values()), default=0)
