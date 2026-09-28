"""SAM 3D Body backbone execution with TensorRT or CUDA ONNX Runtime."""
from __future__ import annotations
import contextlib
import ctypes
import logging
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from fdanyone.errors import AssetError
LOGGER = logging.getLogger("fdanyone")

@contextmanager
def sam3d_imports(root: Path) -> Iterator[None]:
    """Import the pinned checkout with its relative asset paths intact."""

    old_cwd = Path.cwd()
    root_text = str(root)
    already_present = root_text in sys.path
    if not already_present:
        sys.path.insert(0, root_text)
    previous_weights = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.chdir(root)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
                category=UserWarning,
            )
            yield
    finally:
        os.chdir(old_cwd)
        if previous_weights is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_weights
        if not already_present:
            with contextlib.suppress(ValueError):
                sys.path.remove(root_text)


def _preload_cudnn() -> None:
    """Expose the pip CUDA libraries to ONNX Runtime before session creation."""

    try:
        import nvidia.cudnn as cudnn
    except ImportError:
        return
    if cudnn.__file__ is None:
        return
    library = Path(cudnn.__file__).parent / "lib/libcudnn.so.9"
    if library.is_file():
        with contextlib.suppress(OSError):
            ctypes.CDLL(str(library))


class _OnnxRunner:
    backend_name = "cuda-ort"
    precision = "fp32"

    def __init__(self, path: Path, *, device_index: int) -> None:
        _preload_cudnn()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise AssetError("SAM 3D Body acceleration requires `uv sync` with ONNX Runtime.") from exc
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(path.resolve()),
                sess_options=options,
                providers=[
                    ("CUDAExecutionProvider", {"device_id": device_index}),
                    "CPUExecutionProvider",
                ],
            )
        except Exception as exc:
            raise AssetError(f"Could not load SAM 3D Body ONNX model {path}: {exc}") from exc
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise AssetError(
                f"ONNX Runtime could not enable CUDA for {path.name}; active providers are "
                f"{self.session.get_providers()}."
            )
        self.input_names = tuple(value.name for value in self.session.get_inputs())
        self.output_names = tuple(value.name for value in self.session.get_outputs())

    def __call__(self, **values):
        import torch

        feed = {}
        for name in self.input_names:
            value = values[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous().numpy()
            feed[name] = np.ascontiguousarray(value)
        outputs = self.session.run(self.output_names, feed)
        return {
            name: torch.from_numpy(value)
            for name, value in zip(self.output_names, outputs, strict=True)
        }


def _load_inference_runner(
    path: Path,
    *,
    device_index: int,
    backend: str,
    trt_precision: str,
):
    """Resolve a SAM 3D Body runner, with an explicit and observable auto fallback."""

    from fdanyone.trt import load_trt_runner, normalize_motion_backend

    resolved = normalize_motion_backend(backend)
    if resolved in ("auto", "tensorrt"):
        try:
            runner = load_trt_runner(
                path,
                precision=trt_precision,
                device_index=device_index,
            )
            LOGGER.info(
                "Using TensorRT %s for SAM 3D Body model %s",
                trt_precision,
                path.name,
            )
            return runner
        except (AssetError, ImportError, OSError, RuntimeError) as exc:
            if resolved == "tensorrt":
                raise AssetError(
                    f"TensorRT was requested for {path.name}, but its engine could not be loaded: {exc}"
                ) from exc
            LOGGER.warning(
                "TensorRT %s is unavailable for %s (%s); falling back to CUDA ONNX Runtime.",
                trt_precision,
                path.name,
                exc,
            )
    return _OnnxRunner(path, device_index=device_index)


class _OnnxBackbone:
    """Factory namespace used to keep the torch import local to the worker."""

    @staticmethod
    def create(
        runner,
        image_mean,
        image_std,
        *,
        embed_dim: int,
        patch_size: int,
    ):
        import torch

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.runner = runner
                self.embed_dim = self.embed_dims = embed_dim
                self.patch_size = patch_size
                # SAM-3D-Body's mixed-precision setup detects this attribute.
                # It is deliberately non-persistent: the real backbone lives
                # in the pinned ONNX graph rather than this adapter module.
                self.register_buffer("pos_embed", torch.empty(0), persistent=False)
                self.register_buffer(
                    "mean", image_mean.view(1, 3, 1, 1), persistent=False
                )
                self.register_buffer(
                    "std", image_std.view(1, 3, 1, 1), persistent=False
                )

            def forward(self, value, **_kwargs):
                image = (value.float() * self.std + self.mean) * 255.0
                embedding = self.runner(imgs=image)["image_embeddings"]
                return embedding.to(value.device, dtype=value.dtype)

        return Module()


@contextmanager
def _sam3d_onnx_backbone(root: Path, runner):
    """Construct SAM-3D-Body around the exported backbone without Torch Hub."""

    import torch

    sam_root = root
    sam_root_text = str(sam_root)
    already_present = sam_root_text in sys.path
    if not already_present:
        sys.path.insert(0, sam_root_text)

    from sam_3d_body import build_models
    from sam_3d_body.models.meta_arch import sam3d_body as meta_arch

    original_factory = meta_arch.create_backbone
    original_loader = build_models.load_state_dict

    def create_backbone(name, cfg=None):
        if name != "dinov3_vith16plus" or cfg is None:
            raise AssetError(
                f"The pinned SAM-3D-Body export expects dinov3_vith16plus, got {name!r}."
            )
        return _OnnxBackbone.create(
            runner,
            torch.tensor(cfg.MODEL.IMAGE_MEAN, dtype=torch.float32),
            torch.tensor(cfg.MODEL.IMAGE_STD, dtype=torch.float32),
            embed_dim=1280,
            patch_size=16,
        )

    def load_without_backbone(module, state_dict, strict=False, logger=None):
        # The separately exported ONNX graph contains these exact weights.
        # Excluding their PyTorch copies avoids allocating and then discarding
        # the full DINOv3 backbone and prevents an unpinned Torch Hub download.
        filtered = state_dict.copy()
        for key in tuple(filtered):
            if key.startswith("backbone.") or ".backbone." in key:
                del filtered[key]
        checkpoint_logger = original_loader.__globals__.get("log")
        was_disabled = getattr(checkpoint_logger, "disabled", False)
        if checkpoint_logger is not None:
            checkpoint_logger.disabled = True
        try:
            return original_loader(
                module,
                filtered,
                strict=strict,
                logger=logger,
            )
        finally:
            if checkpoint_logger is not None:
                checkpoint_logger.disabled = was_disabled

    meta_arch.create_backbone = create_backbone
    build_models.load_state_dict = load_without_backbone
    try:
        yield
    finally:
        meta_arch.create_backbone = original_factory
        build_models.load_state_dict = original_loader
        if not already_present:
            with contextlib.suppress(ValueError):
                sys.path.remove(sam_root_text)
