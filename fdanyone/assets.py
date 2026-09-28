# Modified for sam3dbx4danyone; project changes use AGPL-3.0-only.
# Inherited 4DAnyone and third-party code retains its terms; see NOTICE.
"""Pinned model assets for SAM 3D Body, BiRefNet and 4DAnyone."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from fdanyone.errors import AssetError

HF_REPO_ID = "AntResearch/4DAnyone"


HF_REVISION = "442816913e7cc75be2ede1a5c93a86d936d032f1"


BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet"


BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


BIREFNET_DIR = "birefnet"


BIREFNET_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
)


CHECKPOINT = "4danyone/model.safetensors"


WAN_VAE = "4danyone/Wan2.2_VAE.pth"


PROMPT_CONTEXT = "4danyone/prompt_context.safetensors"


MODEL_FILES = (
    CHECKPOINT,
    WAN_VAE,
    PROMPT_CONTEXT,
)


EXAMPLE_FILES = (
    "data/source/pexels/10331522-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/15443888_1080_1920_100fps.mp4",
    "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4",
    "data/source/pexels/5385965-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/5390224-uhd_2160_4096_30fps.mp4",
    "data/source/pexels/5390836-uhd_2160_4096_30fps.mp4",
    "data/source/pexels/5435720-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/5885633-hd_1080_1920_25fps.mp4",
    "data/source/pexels/5999210-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/6003989-uhd_2160_3840_30fps.mp4",
    "data/source/pexels/6191453-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/6616344-hd_1080_1920_25fps.mp4",
    "data/source/pexels/6980035-uhd_2160_4096_30fps.mp4",
    "data/source/pexels/7017803-hd_1080_1920_30fps.mp4",
    "data/source/pexels/7080903-hd_1080_1920_30fps.mp4",
    "data/source/pexels/7341232-uhd_2160_3840_25fps.mp4",
    "data/source/pexels/7480858-uhd_2160_3840_25fps.mp4",
    "data/source/pexels/7716891-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/8059623-hd_1080_1920_25fps.mp4",
    "data/source/pexels/8431510-uhd_2160_4096_25fps.mp4",
)


@dataclass(frozen=True)
class BaseAssets:
    vae: Path
    prompt_context: Path


def _require_file(path: Path, label: str, command: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AssetError(f"{label} does not exist: {resolved}. Run `uv run python {command}` to install it.")
    return resolved


def resolve_checkpoint(path: str | Path | None = None, model_dir: str | Path = "models") -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise AssetError(f"Checkpoint override does not exist: {resolved}")
        return resolved
    return _require_file(Path(model_dir) / CHECKPOINT, "Checkpoint", "scripts/download_model.py")


def resolve_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser() / BIREFNET_DIR
    for relative in BIREFNET_FILES:
        _require_file(root / relative, "BiRefNet file", "scripts/download_model.py")
    return root.resolve()


def resolve_base_assets(model_dir: str | Path = "models") -> BaseAssets:
    """Resolve the local VAE and frozen prompt conditioning."""

    root = Path(model_dir).expanduser()
    return BaseAssets(
        vae=_require_file(root / WAN_VAE, "VAE", "scripts/download_model.py"),
        prompt_context=_require_file(root / PROMPT_CONTEXT, "Prompt conditioning", "scripts/download_model.py"),
    )


SAM3D_REPO_URL = "https://github.com/facebookresearch/sam-3d-body.git"
SAM3D_REVISION = "b5c765a0d89d789985e186d396315e7590887b94"
SAM3D_HF_REPO_ID = "nvidia/GEM-X"
SAM3D_HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
SAM3D_DIR = "sam3d-body"
SAM3D_FILES = ("mhr_model.pt", "sam3d_body.ckpt", "model_config.yaml",
               "onnx/sam3db_backbone.onnx", "onnx/sam3db_backbone.onnx.data")

@dataclass(frozen=True)
class Sam3DAssets:
    mhr_model: Path
    sam3d_checkpoint: Path
    sam3d_config: Path
    sam3d_backbone_onnx: Path


def resolve_sam3d_assets(model_dir: str | Path = "models") -> Sam3DAssets:
    root = Path(model_dir).expanduser().resolve() / SAM3D_DIR
    for relative in SAM3D_FILES:
        _require_file(root / relative, "SAM 3D Body asset", "scripts/download_model.py")
    return Sam3DAssets(root / SAM3D_FILES[0], root / SAM3D_FILES[1],
                       root / SAM3D_FILES[2], root / SAM3D_FILES[3])
