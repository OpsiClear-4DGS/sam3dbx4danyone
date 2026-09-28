"""Validate existing SAM 3D Body TensorRT engines against CUDA ONNX Runtime."""

from __future__ import annotations

import argparse
import json

from fdanyone.assets import resolve_sam3d_assets
from fdanyone.trt import (
    SAM3D_ENGINE_SPECS,
    record_engine_validation,
    validate_engine,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(SAM3D_ENGINE_SPECS),
        default=tuple(SAM3D_ENGINE_SPECS),
    )
    return parser.parse_args()


def main() -> None:
    from fdanyone.trt import _configure_trt_environment

    _configure_trt_environment()
    import torch

    from fdanyone.trt import engine_artifact_paths

    args = _arguments()
    assets = resolve_sam3d_assets(args.model_dir)
    sources = {
        "sam3db_backbone": assets.sam3d_backbone_onnx,
    }
    capability = tuple(torch.cuda.get_device_capability(args.device))
    reports = {}
    failed = []
    for name in args.models:
        _, metadata = engine_artifact_paths(
            sources[name],
            precision=args.precision,
            capability=capability,
        )
        report = validate_engine(
            sources[name],
            SAM3D_ENGINE_SPECS[name],
            precision=args.precision,
            device_index=args.device,
        )
        record_engine_validation(metadata, report)
        reports[name] = report
        if not report["accepted"]:
            failed.append(name)
    print(json.dumps(reports, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"TensorRT validation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
