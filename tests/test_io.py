from __future__ import annotations

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fdanyone.io import link_or_copy_file


class LinkOrCopyFileTests(unittest.TestCase):
    def test_uses_hard_link_on_one_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scratch" / "video.mp4"
            destination = root / "result" / "video.mp4"
            source.parent.mkdir()
            source.write_bytes(b"immutable-video")

            linked = link_or_copy_file(source, destination)

            self.assertEqual(linked, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertTrue(os.path.samefile(source, destination))

    def test_falls_back_to_copy_across_filesystems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "nested" / "destination.bin"
            source.write_bytes(b"portable-publication")

            with patch("fdanyone.io.os.link", side_effect=OSError(errno.EXDEV, "cross-device link")):
                copied = link_or_copy_file(source, destination)

            self.assertEqual(copied, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(os.path.samefile(source, destination))

    def test_refuses_to_overwrite_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"new")
            destination.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                link_or_copy_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
