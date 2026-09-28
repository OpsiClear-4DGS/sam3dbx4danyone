"""Strongly typed TensorRT build artifacts and device-native inference."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from fdanyone.errors import AssetError, ConfigurationError

if TYPE_CHECKING:
    from torch import Tensor

TRT_ENGINE_FORMAT = "fdanyone-tensorrt-engine"
TRT_ENGINE_FORMAT_VERSION = 3
TRT_PRECISIONS = ("fp32", "fp16")
TRT_CALIBRATION_POLICY = "fdanyone-gem-x-domains-v1"
TRT_VALIDATION_POLICY = "fdanyone-trt-output-parity-v1"


def _configure_trt_environment() -> None:
    """Keep Myelin's build/runtime math environment bit-identical."""

    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


@dataclass(frozen=True)
class ShapeRange:
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "min": list(self.minimum),
            "opt": list(self.optimum),
            "max": list(self.maximum),
        }


@dataclass(frozen=True)
class EngineSpec:
    name: str
    profiles: dict[str, ShapeRange]


SAM3D_ENGINE_SPECS = {
    "sam3db_backbone": EngineSpec(
        name="sam3db_backbone",
        profiles={
            "imgs": ShapeRange((1, 3, 512, 512), (16, 3, 512, 512), (16, 3, 512, 512)),
        },
    ),

}


def normalize_motion_backend(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"motion_backend must be a string, got {value!r}.")
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "trt": "tensorrt",
        "tensorrt": "tensorrt",
        "ort": "cuda-ort",
        "cuda-ort": "cuda-ort",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise ConfigurationError(
            f"motion_backend must be one of ['auto', 'tensorrt', 'cuda-ort'], got {value!r}."
        ) from None


def normalize_trt_precision(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"motion_precision must be a string, got {value!r}.")
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "full": "fp32",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise ConfigurationError(
            f"motion_precision must be one of {list(TRT_PRECISIONS)}, got {value!r}."
        ) from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_bundle_sha256(path: str | Path) -> str:
    """Hash an ONNX protobuf and its conventional external-data file."""

    source = Path(path).expanduser().resolve()
    files = [source]
    external = Path(f"{source}.data")
    if external.is_file():
        files.append(external)
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def engine_artifact_paths(
    onnx_path: str | Path,
    *,
    precision: str,
    capability: tuple[int, int],
) -> tuple[Path, Path]:
    source = Path(onnx_path).expanduser().resolve()
    if precision not in TRT_PRECISIONS:
        raise ValueError(f"Unknown TensorRT precision {precision!r}.")
    root = source.parent.parent / "trt"
    stem = source.stem
    suffix = f"{precision}.sm{capability[0]}{capability[1]}"
    engine = root / f"{stem}.{suffix}.engine"
    return engine, engine.with_suffix(".engine.json")


def _repair_dynamic_output_metadata(model) -> None:
    """Tie stale fixed batch annotations to the graph's dynamic B axis."""

    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("shape") and tensor_type.shape.dim:
            first = tensor_type.shape.dim[0]
            first.ClearField("dim_value")
            first.dim_param = "B"


def _convert_safe_linear_fp16(model) -> None:
    """Convert only linear compute boundaries to FP16 in one graph pass."""

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    safe_ops = {"Conv", "Gemm", "MatMul"}
    original_nodes = []
    consumers: dict[str, list[object]] = {}
    for original in model.graph.node:
        node = onnx.NodeProto()
        node.CopyFrom(original)
        original_nodes.append(node)
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    converted_initializers: set[str] = set()
    duplicated_initializers = []

    def low_precision_initializer(name: str) -> str:
        initializer = initializers[name]
        if initializer.data_type != onnx.TensorProto.FLOAT:
            return name
        consumers_for_value = consumers.get(name, [])
        shared_with_fp32 = any(node.op_type not in safe_ops for node in consumers_for_value)
        target_name = f"{name}__fdanyone_fp16" if shared_with_fp32 else name
        if target_name in converted_initializers:
            return target_name
        array = numpy_helper.to_array(initializer)
        converted = numpy_helper.from_array(
            np.asarray(array, dtype=np.float16),
            name=target_name,
        )
        if shared_with_fp32:
            duplicated_initializers.append(converted)
        else:
            initializer.CopyFrom(converted)
        converted_initializers.add(target_name)
        return target_name

    rewritten_nodes = []
    for node_index, node in enumerate(original_nodes):
        if node.op_type not in safe_ops:
            rewritten_nodes.append(node)
            continue
        prefix = node.name or f"{node.op_type}_{node_index}"
        before = []
        after = []
        for input_index, input_name in enumerate(tuple(node.input)):
            initializer = initializers.get(input_name)
            if initializer is not None:
                node.input[input_index] = low_precision_initializer(input_name)
                continue
            cast_output = f"{input_name}__fdanyone_fp16_{node_index}_{input_index}"
            before.append(
                helper.make_node(
                    "Cast",
                    [input_name],
                    [cast_output],
                    name=f"{prefix}__input_{input_index}_to_fp16",
                    to=onnx.TensorProto.FLOAT16,
                )
            )
            node.input[input_index] = cast_output
        for output_index, output_name in enumerate(tuple(node.output)):
            low_output = f"{output_name}__fdanyone_fp16"
            node.output[output_index] = low_output
            after.append(
                helper.make_node(
                    "Cast",
                    [low_output],
                    [output_name],
                    name=f"{prefix}__output_{output_index}_to_fp32",
                    to=onnx.TensorProto.FLOAT,
                )
            )
        rewritten_nodes.extend(before)
        rewritten_nodes.append(node)
        rewritten_nodes.extend(after)

    model.graph.ClearField("node")
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(duplicated_initializers)


def _calibration_data(spec: EngineSpec) -> dict[str, object]:
    """Create deterministic samples in each exported model's real input domain."""

    import numpy as np

    generator = np.random.default_rng(20260831)
    if spec.name == "sam3db_backbone":
        # The adapter reverses SAM's normalization and feeds pixel-domain RGB.
        return {
            "imgs": generator.uniform(0.0, 255.0, (1, 3, 512, 512)).astype(
                np.float32
            )
        }
    raise ValueError(f"No TensorRT calibration domain is defined for {spec.name!r}.")


def _prepare_onnx(
    source: Path,
    precision: str,
    destination: Path,
    spec: EngineSpec,
) -> None:
    import onnx

    if precision == "fp32":
        model = onnx.load(str(source), load_external_data=False)
        _repair_dynamic_output_metadata(model)
        onnx.save_model(model, str(destination))
        external = Path(f"{source}.data")
        if external.is_file():
            os.symlink(external, Path(f"{destination}.data"))
        return

    if spec.name == "sam3db_backbone":
        # ModelOpt 0.46 currently fails while reconciling the two nested RoPE
        # If subgraphs in this DINOv3 export.  Use a deliberately conservative
        # graph policy: only GEMM-heavy MatMul/Conv nodes and their weights are
        # FP16; normalization, softmax, reductions, residual arithmetic, and
        # all control flow remain FP32.  The engine validator checks the final
        # outputs against the source graph before this artifact is accepted.
        model = onnx.load(str(source), load_external_data=True)
        _convert_safe_linear_fp16(model)
        _repair_dynamic_output_metadata(model)
        onnx.save_model(
            model,
            str(destination),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{destination.name}.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        return

    try:
        from modelopt.onnx.autocast import convert_to_mixed_precision
        from modelopt.onnx.autocast import referencerunner as modelopt_referencerunner
    except ImportError as exc:
        raise AssetError(
            "FP16 TensorRT builds require `uv sync --extra trt` (NVIDIA ModelOpt)."
        ) from exc
    # ModelOpt 0.46 checks ModelProto.ByteSize() after loading external data.
    # Protobuf raises before returning for graphs above 2 GiB (the SAM backbone and
    # DINOv3 here).  Force the code's intended file-backed reference path so
    # activation-range analysis still runs instead of disabling calibration or
    # applying a blanket FP16 cast.
    original_external_data_check = (
        modelopt_referencerunner.onnx_utils.check_model_uses_external_data
    )
    modelopt_referencerunner.onnx_utils.check_model_uses_external_data = lambda _model: True
    try:
        model = convert_to_mixed_precision(
            onnx_path=str(source),
            low_precision_type="fp16",
            keep_io_types=True,
            calibration_data=_calibration_data(spec),
            providers=["cpu"],
            opset=19,
        )
    finally:
        modelopt_referencerunner.onnx_utils.check_model_uses_external_data = (
            original_external_data_check
        )
    _repair_dynamic_output_metadata(model)
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{destination.name}.data",
        size_threshold=1024,
        convert_attribute=False,
    )


def build_engine(
    onnx_path: str | Path,
    spec: EngineSpec,
    *,
    precision: str,
    device_index: int = 0,
    workspace_bytes: int = 2 << 30,
) -> tuple[Path, Path]:
    """Build and atomically publish one strongly typed TensorRT engine."""

    _configure_trt_environment()

    import tensorrt as trt
    import torch

    if precision not in TRT_PRECISIONS:
        raise ConfigurationError(f"precision must be one of {TRT_PRECISIONS}, got {precision!r}.")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive.")
    source = Path(onnx_path).expanduser().resolve()
    if not source.is_file():
        raise AssetError(f"ONNX source does not exist: {source}")
    torch.cuda.set_device(device_index)
    capability = tuple(torch.cuda.get_device_capability(device_index))
    engine_path, metadata_path = engine_artifact_paths(
        source,
        precision=precision,
        capability=capability,
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    with TemporaryDirectory(prefix=f".{spec.name}-{precision}-", dir=engine_path.parent) as temporary:
        temporary_root = Path(temporary)
        prepared = temporary_root / source.name
        _prepare_onnx(source, precision, prepared, spec)

        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(prepared)):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise AssetError(f"TensorRT could not parse {source.name}:\n{errors}")
        network_inputs = {network.get_input(index).name for index in range(network.num_inputs)}
        if network_inputs != set(spec.profiles):
            raise AssetError(
                f"TensorRT profile for {spec.name} has inputs {sorted(spec.profiles)}, "
                f"but the graph has {sorted(network_inputs)}."
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        # TensorRT enables TF32 tactics by default for FP32 graphs.  Keep the
        # reference engine genuinely FP32 so it can serve as a strict parity
        # oracle for the explicitly converted FP16 graph.
        if precision == "fp32":
            config.clear_flag(trt.BuilderFlag.TF32)
        profile = builder.create_optimization_profile()
        for name, shape_range in spec.profiles.items():
            profile.set_shape(name, shape_range.minimum, shape_range.optimum, shape_range.maximum)
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise AssetError(f"TensorRT failed to build {spec.name} ({precision}).")
        serialized_bytes = bytes(serialized)
        temporary_engine = temporary_root / engine_path.name
        temporary_engine.write_bytes(serialized_bytes)

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_bytes)
        if engine is None:
            raise AssetError(f"TensorRT built but could not deserialize {spec.name}.")
        io_tensors = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            io_tensors.append(
                {
                    "name": name,
                    "mode": str(engine.get_tensor_mode(name)).rsplit(".", 1)[-1].lower(),
                    "dtype": str(engine.get_tensor_dtype(name)).rsplit(".", 1)[-1].lower(),
                    "shape": list(engine.get_tensor_shape(name)),
                }
            )
        metadata = {
            "format": TRT_ENGINE_FORMAT,
            "format_version": TRT_ENGINE_FORMAT_VERSION,
            "model": spec.name,
            "source_onnx": source.name,
            "source_bundle_sha256": model_bundle_sha256(source),
            "precision": precision,
            "precision_policy": (
                "fp32-tf32-disabled"
                if precision == "fp32"
                else (
                    "safe-linear-fp16-fp32-io"
                    if spec.name == "sam3db_backbone"
                    else "modelopt-autocast-fp16-fp32-io"
                )
            ),
            "strongly_typed": True,
            "nvidia_tf32_override": os.environ["NVIDIA_TF32_OVERRIDE"],
            "calibration_policy": (
                None if precision == "fp32" else TRT_CALIBRATION_POLICY
            ),
            "tensorrt_version": trt.__version__,
            "cuda_capability": list(capability),
            "workspace_bytes": workspace_bytes,
            "profiles": {name: value.to_dict() for name, value in spec.profiles.items()},
            "io_tensors": io_tensors,
            "engine_bytes": len(serialized_bytes),
            "engine_sha256": hashlib.sha256(serialized_bytes).hexdigest(),
            "device_memory_bytes": int(engine.device_memory_size_v2),
        }
        temporary_metadata = temporary_root / metadata_path.name
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_engine, engine_path)
        os.replace(temporary_metadata, metadata_path)
    return engine_path, metadata_path


def _torch_dtype(trt, dtype):
    import torch

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    try:
        return mapping[dtype]
    except KeyError:
        raise AssetError(f"Unsupported TensorRT tensor dtype: {dtype}") from None


def _validation_thresholds(precision: str) -> dict[str, float]:
    if precision == "fp32":
        return {"minimum_cosine_similarity": 0.9999, "maximum_relative_rmse": 0.01}
    return {"minimum_cosine_similarity": 0.999, "maximum_relative_rmse": 0.03}


def _parity_metrics(reference, candidate) -> dict[str, float | bool]:
    import torch

    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    difference = reference - candidate
    reference_rms = float(reference.square().mean().sqrt())
    rmse = float(difference.square().mean().sqrt())
    reference64 = reference.double().flatten()
    candidate64 = candidate.double().flatten()
    denominator = float(reference64.norm() * candidate64.norm())
    cosine = (
        1.0
        if denominator == 0.0 and bool(torch.equal(reference, candidate))
        else (0.0 if denominator == 0.0 else float(reference64.dot(candidate64)) / denominator)
    )
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": rmse / max(reference_rms, 1e-12),
        "cosine_similarity": cosine,
    }


def validate_engine(
    onnx_path: str | Path,
    spec: EngineSpec,
    *,
    precision: str,
    device_index: int = 0,
) -> dict[str, object]:
    """Compare a built engine with CUDA ORT at every distinct profile batch."""

    _configure_trt_environment()

    import torch

    from fdanyone.motion.sam3d_runtime import _OnnxRunner

    source = Path(onnx_path).expanduser().resolve()
    calibration = _calibration_data(spec)
    batch_sizes = sorted(
        {
            shape[0]
            for value in spec.profiles.values()
            for shape in (value.minimum, value.optimum, value.maximum)
        }
    )
    reference_runner = _OnnxRunner(source, device_index=device_index)
    candidate_runner = load_trt_runner(
        source,
        precision=precision,
        device_index=device_index,
        require_validated=False,
    )
    thresholds = _validation_thresholds(precision)
    cases = []
    accepted = True
    try:
        for batch_size in batch_sizes:
            values = {}
            for name, value in calibration.items():
                tensor = torch.from_numpy(value)
                if tensor.shape[0] != batch_size:
                    tensor = tensor.repeat(batch_size, *(1 for _ in tensor.shape[1:]))
                values[name] = tensor

            reference_started = time.perf_counter()
            reference = reference_runner(**values)
            reference_seconds = time.perf_counter() - reference_started
            candidate_runner(**values)
            torch.cuda.synchronize(device_index)
            candidate_started = time.perf_counter()
            candidate = {
                name: value.cpu() for name, value in candidate_runner(**values).items()
            }
            torch.cuda.synchronize(device_index)
            candidate_seconds = time.perf_counter() - candidate_started

            outputs = {}
            case_accepted = True
            for name in reference:
                metrics = _parity_metrics(reference[name], candidate[name])
                output_accepted = bool(
                    metrics["finite"]
                    and metrics["cosine_similarity"]
                    >= thresholds["minimum_cosine_similarity"]
                    and metrics["relative_rmse"]
                    <= thresholds["maximum_relative_rmse"]
                )
                metrics["accepted"] = output_accepted
                outputs[name] = metrics
                case_accepted &= output_accepted
            cases.append(
                {
                    "batch_size": batch_size,
                    "reference_seconds": reference_seconds,
                    "tensorrt_seconds": candidate_seconds,
                    "accepted": case_accepted,
                    "outputs": outputs,
                }
            )
            accepted &= case_accepted
    finally:
        del reference_runner, candidate_runner
        torch.cuda.empty_cache()
    return {
        "policy": TRT_VALIDATION_POLICY,
        "calibration_policy": TRT_CALIBRATION_POLICY,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "precision": precision,
        "thresholds": thresholds,
        "accepted": accepted,
        "cases": cases,
    }


def record_engine_validation(metadata_path: str | Path, validation: dict[str, object]) -> None:
    path = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(path.read_text())
    metadata["validation"] = validation
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class TensorRTRunner:
    """Execute one locally built engine on Torch-owned CUDA buffers."""

    backend_name = "tensorrt"

    def __init__(
        self,
        engine_path: str | Path,
        metadata_path: str | Path,
        *,
        source_onnx: str | Path,
        device_index: int,
        expected_precision: str,
        require_validated: bool = True,
    ) -> None:
        _configure_trt_environment()

        import tensorrt as trt
        import torch

        self.engine_path = Path(engine_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        if not self.engine_path.is_file() or not self.metadata_path.is_file():
            raise AssetError(f"TensorRT engine artifacts are incomplete for {self.engine_path.name}.")
        metadata = json.loads(self.metadata_path.read_text())
        if (metadata.get("format"), metadata.get("format_version")) != (
            TRT_ENGINE_FORMAT,
            TRT_ENGINE_FORMAT_VERSION,
        ):
            raise AssetError(f"Unknown TensorRT engine metadata format: {self.metadata_path}")
        if metadata.get("precision") != expected_precision:
            raise AssetError(
                f"TensorRT engine precision is {metadata.get('precision')!r}, "
                f"expected {expected_precision!r}: {self.metadata_path}"
            )
        validation = metadata.get("validation")
        if require_validated and (
            not isinstance(validation, dict)
            or validation.get("policy") != TRT_VALIDATION_POLICY
            or validation.get("accepted") is not True
        ):
            raise AssetError(
                f"TensorRT engine has no passing {TRT_VALIDATION_POLICY} record: "
                f"{self.metadata_path}. Run `uv run --extra trt python "
                "scripts/validate_trt_engines.py`."
            )
        capability = tuple(torch.cuda.get_device_capability(device_index))
        checks = {
            "tensorrt_version": trt.__version__,
            "cuda_capability": list(capability),
            "nvidia_tf32_override": os.environ["NVIDIA_TF32_OVERRIDE"],
            "source_bundle_sha256": model_bundle_sha256(source_onnx),
            "engine_sha256": _sha256_file(self.engine_path),
        }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in checks.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise AssetError(f"TensorRT engine cache is stale or incompatible: {mismatched}")

        self.device_index = device_index
        self.device = torch.device("cuda", device_index)
        self.metadata = metadata
        self.precision = str(metadata["precision"])
        self._trt = trt
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise AssetError(f"Could not deserialize TensorRT engine {self.engine_path}.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise AssetError(f"Could not create TensorRT context for {self.engine_path}.")
        self.input_names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        )
        self.output_names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        )
        self._outputs: dict[tuple[str, tuple[int, ...]], Tensor] = {}

    def __call__(self, **values):
        import numpy as np
        import torch

        if set(values) != set(self.input_names):
            raise ValueError(
                f"TensorRT inputs for {self.engine_path.name} must be {self.input_names}, got {tuple(values)}."
            )
        inputs = {}
        for name in self.input_names:
            value = values[name]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(np.ascontiguousarray(value))
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TensorRT input {name!r} must be a Torch tensor or NumPy array.")
            expected_dtype = _torch_dtype(self._trt, self.engine.get_tensor_dtype(name))
            tensor = value.detach().to(device=self.device, dtype=expected_dtype).contiguous()
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise AssetError(f"TensorRT rejected input shape {tuple(tensor.shape)} for {name!r}.")
            inputs[name] = tensor
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        unresolved = self.context.infer_shapes()
        if unresolved:
            raise AssetError(f"TensorRT could not infer shapes for tensors {tuple(unresolved)}.")
        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if not shape or any(dimension <= 0 for dimension in shape):
                raise AssetError(f"TensorRT produced unresolved output shape {shape} for {name!r}.")
            key = (name, shape)
            expected_dtype = _torch_dtype(self._trt, self.engine.get_tensor_dtype(name))
            output = self._outputs.get(key)
            if output is None or output.dtype != expected_dtype:
                output = torch.empty(shape, dtype=expected_dtype, device=self.device)
                self._outputs[key] = output
            self.context.set_tensor_address(name, int(output.data_ptr()))
            outputs[name] = output

        stream = torch.cuda.current_stream(self.device_index)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise AssetError(f"TensorRT execution failed for {self.engine_path.name}.")
        # Inputs remain live until this call returns, and allocator reuse on the
        # same stream is ordered after TensorRT's enqueued work. Consumers can
        # therefore continue asynchronously without a per-call synchronization.
        return outputs


def load_trt_runner(
    onnx_path: str | Path,
    *,
    precision: str,
    device_index: int,
    require_validated: bool = True,
) -> TensorRTRunner:
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    engine, metadata = engine_artifact_paths(
        onnx_path,
        precision=precision,
        capability=capability,
    )
    return TensorRTRunner(
        engine,
        metadata,
        source_onnx=onnx_path,
        device_index=device_index,
        expected_precision=precision,
        require_validated=require_validated,
    )


__all__ = [
    "EngineSpec",
    "SAM3D_ENGINE_SPECS",
    "ShapeRange",
    "TensorRTRunner",
    "build_engine",
    "engine_artifact_paths",
    "load_trt_runner",
    "model_bundle_sha256",
    "normalize_motion_backend",
    "normalize_trt_precision",
    "record_engine_validation",
    "validate_engine",
]
