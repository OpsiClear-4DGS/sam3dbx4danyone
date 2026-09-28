from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fdanyone.errors import ConfigurationError
from fdanyone.trt import (
    SAM3D_ENGINE_SPECS,
    _calibration_data,
    engine_artifact_paths,
    model_bundle_sha256,
    normalize_motion_backend,
    normalize_trt_precision,
)


class TensorRTArtifactTests(unittest.TestCase):
    def test_backend_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_motion_backend("TRT"), "tensorrt")
        self.assertEqual(normalize_motion_backend("ort"), "cuda-ort")
        with self.assertRaisesRegex(ConfigurationError, "motion_backend"):
            normalize_motion_backend("cpu")

    def test_bundle_hash_includes_external_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            onnx = Path(temporary) / "model.onnx"
            data = Path(f"{onnx}.data")
            onnx.write_bytes(b"graph")
            data.write_bytes(b"weights-a")
            first = model_bundle_sha256(onnx)
            data.write_bytes(b"weights-b")
            second = model_bundle_sha256(onnx)
        self.assertNotEqual(first, second)

    def test_precision_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_trt_precision("half"), "fp16")
        self.assertEqual(normalize_trt_precision("float32"), "fp32")
        with self.assertRaisesRegex(ConfigurationError, "motion_precision"):
            normalize_trt_precision("int8")

    def test_engine_path_is_precision_and_architecture_specific(self) -> None:
        engine, metadata = engine_artifact_paths(
            "/models/sam3d-body/onnx/sam3db_backbone.onnx",
            precision="fp16",
            capability=(8, 9),
        )
        self.assertEqual(engine.name, "sam3db_backbone.fp16.sm89.engine")
        self.assertEqual(metadata.name, "sam3db_backbone.fp16.sm89.engine.json")

    def test_sam_profile_and_calibration_match_runtime_crops(self) -> None:
        spec = SAM3D_ENGINE_SPECS["sam3db_backbone"]
        self.assertEqual(spec.profiles["imgs"].minimum, (1, 3, 512, 512))
        self.assertEqual(spec.profiles["imgs"].maximum, (16, 3, 512, 512))
        calibration = _calibration_data(spec)["imgs"]
        self.assertEqual(calibration.shape, (1, 3, 512, 512))
        self.assertGreaterEqual(float(calibration.min()), 0.0)
        self.assertLessEqual(float(calibration.max()), 255.0)


if __name__ == "__main__":
    unittest.main()
