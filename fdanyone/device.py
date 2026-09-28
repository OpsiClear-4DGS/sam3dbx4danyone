# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""CUDA device selection shared by pipeline and isolated workers."""

from __future__ import annotations

import os
from collections.abc import MutableMapping, Sequence

from fdanyone.errors import ConfigurationError

CUDA_ALLOCATOR_CONF = "PYTORCH_CUDA_ALLOC_CONF"
CUDA_MAX_SPLIT_SIZE_MB = 4096


def default_cpu_threads(workers: int = 1) -> int:
    """Budget CPU tensor work across GPU workers without serializing each job."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(8, available // max(1, workers)))


def configure_inference_cpu_environment() -> None:
    threads = (os.environ.get('OMP_NUM_THREADS') or os.environ.get('MKL_NUM_THREADS')
               or str(default_cpu_threads()))
    os.environ.setdefault('OMP_NUM_THREADS', threads)
    os.environ.setdefault('MKL_NUM_THREADS', threads)


def configure_inference_cuda_allocator(
    environment: MutableMapping[str, str] | None = None,
    *,
    max_split_size_mb: int = CUDA_MAX_SPLIT_SIZE_MB,
    expandable_segments: bool = False,
    garbage_collection_threshold: float | None = None,
) -> None:
    """Configure the CUDA allocator before PyTorch creates a CUDA context."""

    if max_split_size_mb <= 0:
        raise ValueError(f"max_split_size_mb must be positive, got {max_split_size_mb}.")
    if garbage_collection_threshold is not None and not 0 < garbage_collection_threshold < 1:
        raise ValueError(
            "garbage_collection_threshold must be between zero and one, "
            f"got {garbage_collection_threshold}."
        )
    environment = os.environ if environment is None else environment
    current = environment.get(CUDA_ALLOCATOR_CONF, "").strip()
    options = [option.strip() for option in current.split(",") if option.strip()]
    names = {option.partition(":")[0] for option in options}
    if "max_split_size_mb" not in names:
        options.append(f"max_split_size_mb:{max_split_size_mb}")
    if expandable_segments and "expandable_segments" not in names:
        options.append("expandable_segments:True")
    if garbage_collection_threshold is not None and "garbage_collection_threshold" not in names:
        options.append(f"garbage_collection_threshold:{garbage_collection_threshold:g}")
    environment[CUDA_ALLOCATOR_CONF] = ",".join(options)


def select_cuda_device(device: str) -> tuple[str, int]:
    """Validate, select, and normalize one CUDA device."""

    import torch

    try:
        requested = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid CUDA device {device!r}.") from exc
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise ConfigurationError(f"4DAnyone requires an available CUDA device, got {device!r}.")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ConfigurationError(
            f"CUDA device index {index} is unavailable; visible device count is {torch.cuda.device_count()}."
        )
    torch.cuda.set_device(index)
    return f"cuda:{index}", index


def select_cuda_devices(gpu_ids: Sequence[int] | None = None) -> tuple[str, ...]:
    """Select an ordered set of CUDA-visible devices for one inference run."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise ConfigurationError("4DAnyone requires at least one available CUDA device.")

    if gpu_ids is None:
        selected = tuple(range(torch.cuda.device_count()))
    else:
        if isinstance(gpu_ids, (str, bytes)) or not isinstance(gpu_ids, Sequence) or not gpu_ids:
            raise ConfigurationError("gpu_ids must be a non-empty list of CUDA-visible device IDs.")
        selected = tuple(gpu_ids)
        invalid = [gpu_id for gpu_id in selected if isinstance(gpu_id, bool) or not isinstance(gpu_id, int)]
        if invalid:
            raise ConfigurationError(f"gpu_ids must contain only integers, got {invalid!r}.")
        if len(set(selected)) != len(selected):
            raise ConfigurationError(f"gpu_ids must not contain duplicates, got {list(selected)!r}.")

    available = torch.cuda.device_count()
    unavailable = [gpu_id for gpu_id in selected if gpu_id < 0 or gpu_id >= available]
    if unavailable:
        raise ConfigurationError(
            f"gpu_ids contains unavailable CUDA-visible device IDs {unavailable}; visible device count is {available}."
        )
    torch.cuda.set_device(selected[0])
    return tuple(f"cuda:{gpu_id}" for gpu_id in selected)
