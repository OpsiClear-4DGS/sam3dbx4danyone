# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Download only the active pipeline's pinned assets."""
from __future__ import annotations
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from fdanyone.assets import (
    HF_REPO_ID, HF_REVISION, BIREFNET_FILES, BIREFNET_DIR, BIREFNET_REPO_ID,
    BIREFNET_REVISION, MODEL_FILES, EXAMPLE_FILES,
    SAM3D_DIR, SAM3D_FILES, SAM3D_REPO_URL, SAM3D_REVISION,
    SAM3D_HF_REPO_ID, SAM3D_HF_REVISION, resolve_sam3d_assets,
)
from fdanyone.errors import AssetError
LOGGER = logging.getLogger("fdanyone")

def _snapshot(
    allow_patterns: list[str],
    local_dir: Path,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AssetError("Run `uv sync` before downloading assets.") from exc

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
        )
    except Exception as exc:
        raise AssetError(
            f"Could not download {repo_id}@{revision}. Check the network connection and Hugging Face access."
        ) from exc


def ensure_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser().resolve() / BIREFNET_DIR
    missing = [relative for relative in BIREFNET_FILES if not (root / relative).is_file()]
    if missing:
        LOGGER.info("Downloading BiRefNet foreground model (first run only)")
        _snapshot(
            missing,
            root,
            repo_id=BIREFNET_REPO_ID,
            revision=BIREFNET_REVISION,
        )
    return root


def ensure_models(
    model_dir: str | Path = "models",
) -> Path:
    """Download the first-party generator, foreground assets."""

    models = Path(model_dir).expanduser().resolve()
    missing = [relative for relative in MODEL_FILES if not (models / relative).is_file()]
    if missing:
        LOGGER.info("Downloading %d model files from %s (first run only)", len(missing), HF_REPO_ID)
        # Repository paths match the local layout, so download straight into
        # place; huggingface_hub stages and resumes partial files itself.
        _snapshot(missing, models)
    ensure_foreground_model(models)
    return models


def download_example(data_dir: str = "data") -> dict[str, str]:
    """Download the bundled example clips."""

    data = Path(data_dir).expanduser().resolve()
    destinations = {relative: data / Path(relative).relative_to("data") for relative in EXAMPLE_FILES}
    missing = [relative for relative, destination in destinations.items() if not destination.is_file()]
    if missing:
        # Repository paths carry a leading ``data/`` prefix while --data_dir is
        # the local root itself, so stage the snapshot and move each file.
        staging = data / ".download"
        _snapshot(missing, staging)
        for relative in missing:
            destination = destinations[relative]
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(destination)
        shutil.rmtree(staging)
    return {"examples": str(data / "source/pexels"), "revision": HF_REVISION}


def ensure_example_video(video_path: str | Path) -> Path:
    """Fetch a bundled example clip when its expected file is missing."""

    path = Path(video_path).expanduser()
    if path.is_file():
        return path
    matches = [relative for relative in EXAMPLE_FILES if PurePosixPath(relative).name == path.name]
    if not matches:
        raise AssetError(f"Input video does not exist: {path.resolve()}")
    LOGGER.info("Downloading the bundled example clip %s", path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final rename stays on one filesystem.
    staging = path.parent / ".download"
    _snapshot(matches[:1], staging)
    (staging / matches[0]).replace(path)
    shutil.rmtree(staging)
    return path


def ensure_sam3d(model_dir: str | Path = "models") -> Path:
    models = Path(model_dir).expanduser().resolve()
    root = models / SAM3D_DIR
    missing = [name for name in SAM3D_FILES if not (root / name).is_file()]
    if missing:
        _snapshot(missing, root, repo_id=SAM3D_HF_REPO_ID, revision=SAM3D_HF_REVISION)
    source = root / "source"
    if not source.exists():
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".sam3d-source-", dir=root))
        try:
            subprocess.run(["git", "clone", "--no-checkout", SAM3D_REPO_URL, str(temporary)], check=True)
            subprocess.run(["git", "-C", str(temporary), "checkout", "--detach", SAM3D_REVISION], check=True)
            temporary.rename(source)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SAM3D_REVISION or not (source / "sam_3d_body/__init__.py").is_file():
        raise AssetError(f"Expected complete SAM 3D Body source at revision {SAM3D_REVISION}: {source}")
    resolve_sam3d_assets(models)
    return source


def ensure_turbo(model_dir: str | Path = "models") -> Path:
    from fdanyone.model.turbo_lora import TURBO_FILENAME, TURBO_HF_REVISION, validate_turbo_adapter
    path = Path(model_dir).expanduser().resolve() / TURBO_FILENAME
    if not path.is_file():
        _snapshot([TURBO_FILENAME], Path(model_dir).resolve(), revision=TURBO_HF_REVISION)
    validate_turbo_adapter(path)
    return path


def download_model(model_dir: str = "models", turbo: bool = False) -> dict:
    models = ensure_models(model_dir)
    source = ensure_sam3d(models)
    if turbo:
        ensure_turbo(models)
    return {"models": str(models), "sam3d_source": str(source), "sam3d_revision": SAM3D_REVISION}
