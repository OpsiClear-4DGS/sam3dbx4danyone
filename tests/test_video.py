from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from fdanyone.video import read_rgb_video, write_video


class LosslessVideoTests(unittest.TestCase):
    def test_rgb_h264_veryfast_round_trips_every_byte(self) -> None:
        generator = np.random.default_rng(7)
        frames = tuple(
            generator.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
            for _ in range(3)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lossless.mp4"
            write_video(
                frames,
                path,
                Fraction(25, 1),
                preset="veryfast",
                lossless_rgb=True,
            )

            decoded = read_rgb_video(path, expected_frames=len(frames))

        for actual, expected in zip(decoded, frames, strict=True):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
