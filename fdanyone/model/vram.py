"""Process-level GPU-memory sampling through NVIDIA NVML."""

from __future__ import annotations

import os
from threading import Event, Thread
from types import ModuleType


class ProcessVramMonitor:
    """Sample one process's memory on one physical GPU.

    PyTorch allocator metrics do not include allocations owned by ONNX Runtime,
    TensorRT, CUDA libraries, or video codecs. NVML's per-process accounting is
    therefore the authoritative peak used by the low-VRAM acceptance gate.
    """

    def __init__(
        self,
        device_index: int,
        *,
        interval_seconds: float = 0.05,
        pid: int | None = None,
    ) -> None:
        if device_index < 0:
            raise ValueError(f"device_index must be non-negative, got {device_index}.")
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}.")
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.pid = os.getpid() if pid is None else pid
        self.peak_bytes = 0
        self._nvml: ModuleType | None = None
        self._handle = None
        self._stop = Event()
        self._thread: Thread | None = None
        self._error: BaseException | None = None

    def _resolve_handle(self):
        import torch

        if self._nvml is None:
            raise RuntimeError("NVML was not initialized.")
        uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)
        if not uuid.startswith("GPU-"):
            uuid = f"GPU-{uuid}"
        try:
            return self._nvml.nvmlDeviceGetHandleByUUID(uuid)
        except self._nvml.NVMLError:
            # Old NVML releases may not accept UUID strings. The normal
            # non-remapped index remains a useful fallback for those systems.
            return self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _sample(self) -> None:
        if self._nvml is None or self._handle is None:
            raise RuntimeError("NVML monitor is not running.")
        observed: list[int] = []
        process_queries = (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
            "nvmlDeviceGetMPSComputeRunningProcesses",
        )
        for query_name in process_queries:
            query = getattr(self._nvml, query_name, None)
            if query is None:
                continue
            try:
                processes = query(self._handle)
            except self._nvml.NVMLError_NotSupported:
                continue
            for process in processes:
                if int(process.pid) != self.pid:
                    continue
                used = getattr(process, "usedGpuMemory", None)
                if used is None:
                    continue
                used = int(used)
                # NVML_VALUE_NOT_AVAILABLE is an unsigned all-ones sentinel.
                if 0 <= used < 2**63:
                    observed.append(used)
        if observed:
            # A process can appear in more than one NVML process class. Every
            # record already represents its total allocation, so do not sum.
            self.peak_bytes = max(self.peak_bytes, *observed)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except BaseException as exc:  # surfaced synchronously by stop()
                self._error = exc
                self._stop.set()

    def start(self) -> ProcessVramMonitor:
        if self._thread is not None:
            raise RuntimeError("NVML monitor was started more than once.")
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("Process VRAM measurement requires the `nvidia-ml-py` package.") from exc
        self._nvml = pynvml
        self._nvml.nvmlInit()
        try:
            self._handle = self._resolve_handle()
            self._sample()
            self._thread = Thread(
                target=self._sample_until_stopped,
                name=f"nvml-cuda-{self.device_index}",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._nvml.nvmlShutdown()
            self._nvml = None
            self._handle = None
            raise
        return self

    def stop(self) -> int:
        if self._thread is None or self._nvml is None:
            raise RuntimeError("NVML monitor is not running.")
        self._stop.set()
        self._thread.join()
        try:
            self._sample()
        finally:
            self._nvml.nvmlShutdown()
            self._thread = None
            self._nvml = None
            self._handle = None
        if self._error is not None:
            raise RuntimeError("NVML process-memory sampling failed.") from self._error
        return self.peak_bytes
