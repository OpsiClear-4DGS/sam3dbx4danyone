# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Filesystem helpers for crash-safe result publication."""

from __future__ import annotations

import errno
import json
import os
import shutil
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path

from fdanyone.errors import FourDAnyoneError

_RETRYABLE_TREE_ERRORS = {errno.EBUSY, errno.ENOTEMPTY, errno.ESTALE}
_LINK_FALLBACK_ERRORS = {
    errno.EACCES,
    errno.EMLINK,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.EPERM,
    errno.EXDEV,
}


def write_json(path: str | Path, value: object, *, sort_keys: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=sort_keys) + "\n")
    os.replace(temporary, target)


def link_or_copy_file(source: str | Path, destination: str | Path) -> Path:
    """Publish an immutable file without copying when both paths share a filesystem."""

    source_path = Path(source).expanduser().resolve(strict=True)
    expanded_destination = Path(destination).expanduser()
    destination_path = (
        expanded_destination.parent.resolve() / expanded_destination.name
    )
    if not source_path.is_file():
        raise FourDAnyoneError(f"Publication source is not a regular file: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, destination_path)
    except OSError as exc:
        if exc.errno not in _LINK_FALLBACK_ERRORS:
            raise
        shutil.copy2(source_path, destination_path)
    return destination_path


def remove_tree(
    path: str | Path,
    *,
    attempts: int = 8,
    initial_delay_seconds: float = 0.1,
    ignore_errors: bool = False,
) -> None:
    """Remove a tree, tolerating short directory-entry lag on network filesystems."""

    target = Path(path)
    if attempts <= 0:
        raise ValueError(f"attempts must be positive, got {attempts}.")
    for attempt in range(attempts):
        try:
            shutil.rmtree(target)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            retryable = exc.errno in _RETRYABLE_TREE_ERRORS and attempt + 1 < attempts
            if not retryable:
                if ignore_errors:
                    return
                raise
            time.sleep(initial_delay_seconds * (2**attempt))


class AtomicResultDirectory(AbstractContextManager[Path]):
    """Build beside the destination and rename only after all validation passes."""

    def __init__(self, destination: str | Path):
        expanded = Path(destination).expanduser()
        # Resolve the parent for a stable absolute location, but preserve the
        # leaf itself so a dangling output symlink cannot be followed and
        # mistaken for a nonexistent destination.
        self.destination = expanded.parent.resolve() / expanded.name
        self.working = self.destination.with_name(f".{self.destination.name}.work-{uuid.uuid4().hex[:10]}")
        self._committed = False

    def _destination_exists(self) -> bool:
        return os.path.lexists(self.destination)

    def __enter__(self) -> Path:
        if self._destination_exists():
            raise FourDAnyoneError(
                f"Output directory already exists: {self.destination}. Choose a new directory to avoid mixed runs."
            )
        self.working.mkdir(parents=True)
        return self.working

    def commit(self) -> Path:
        if self._committed:
            return self.destination
        if self._destination_exists():
            raise FourDAnyoneError(
                f"Output directory appeared during inference: {self.destination}. Refusing to overwrite it."
            )
        os.replace(self.working, self.destination)
        self._committed = True
        return self.destination

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            try:
                self.commit()
            except BaseException:
                remove_tree(self.working, ignore_errors=True)
                raise
        else:
            remove_tree(self.working, ignore_errors=True)
        return False
