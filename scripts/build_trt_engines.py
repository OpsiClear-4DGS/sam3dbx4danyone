"""Build local strongly typed TensorRT engines for SAM 3D Body ONNX assets."""

from __future__ import annotations

import argparse
import json

from fdanyone.assets import resolve_sam3d_assets
from fdanyone.trt import (
    SAM3D_ENGINE_SPECS,
    build_engine,
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
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Build only; unvalidated engines are deliberately rejected by runtime loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.workspace_gib <= 0:
        raise SystemExit("--workspace-gib must be positive")
    assets = resolve_sam3d_assets(args.model_dir)
    sources = {
        "sam3db_backbone": assets.sam3d_backbone_onnx,
    }
    outputs = {}
    for name in args.models:
        engine, metadata = build_engine(
            sources[name],
            SAM3D_ENGINE_SPECS[name],
            precision=args.precision,
            device_index=args.device,
            workspace_bytes=int(args.workspace_gib * 2**30),
        )
        outputs[name] = {"engine": str(engine), "metadata": str(metadata)}
        if not args.skip_validation:
            validation = validate_engine(
                sources[name],
                SAM3D_ENGINE_SPECS[name],
                precision=args.precision,
                device_index=args.device,
            )
            record_engine_validation(metadata, validation)
            outputs[name]["validation"] = validation
            if not validation["accepted"]:
                raise SystemExit(f"TensorRT validation failed for {name} ({args.precision}).")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
