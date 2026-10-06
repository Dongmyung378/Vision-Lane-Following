"""Verify archive integrity and the Windows extraction boundary without downloads."""

import hashlib
import io
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import install_carla_windows as installer


class InstallerTests(unittest.TestCase):
    def test_sha256_reads_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.zip"
            path.write_bytes(b"known package bytes")
            self.assertEqual(installer.digest_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_regular_nested_member_stays_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                installer.safe_destination(root, zipfile.ZipInfo("CarlaUE4/Content/map.pak")),
                (root / "CarlaUE4/Content/map.pak").resolve(),
            )

    def test_rejects_traversal_absolute_and_windows_ambiguous_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside", "folder/../../outside", "/absolute", "C:/outside", "folder\\..\\outside", "folder/file:stream", "folder. /file", "folder./file", "folder /file", "CON", "folder/NUL.txt", "folder/file?.txt", "folder/file\x01.txt"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    installer.safe_destination(root, zipfile.ZipInfo(name))

    def test_rejects_zip_symlink(self):
        member = zipfile.ZipInfo("link")
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            installer.safe_destination(Path(directory), member)

    def test_existing_wrong_length_is_not_overwritten_or_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "carla.zip"
            archive.write_bytes(b"unexpected")
            with patch.object(installer.urllib.request, "urlopen") as request:
                with self.assertRaises(ValueError):
                    installer.download("https://example.invalid/package", archive, 1234)
            request.assert_not_called()
            self.assertEqual(archive.read_bytes(), b"unexpected")

    def test_incomplete_partial_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "carla.zip"
            partial = archive.with_suffix(".zip.part")
            partial.write_bytes(b"partial")
            with patch.object(installer.urllib.request, "urlopen") as request:
                with self.assertRaises(ValueError):
                    installer.download("https://example.invalid/package", archive, 1234)
            request.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"partial")

    def test_rejected_response_does_not_leave_an_empty_partial(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length": "1"}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "carla.zip"
            with patch.object(installer.urllib.request, "urlopen", return_value=Response(b"x")):
                with self.assertRaisesRegex(ValueError, "pinned length"):
                    installer.download("https://example.invalid/package", archive, 1234)
            self.assertFalse(archive.with_suffix(".zip.part").exists())

    def test_checksum_mismatch_stops_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "carla").mkdir()
            archive = root / "carla/CARLA_0.9.16.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("CarlaUE4.exe", b"test")
            manifest = {"carla_version": "0.9.16", "windows_package": {
                "url": "https://example.invalid/package", "archive_bytes": archive.stat().st_size,
                "sha256": "0" * 64, "install_directory": "carla/CARLA_0.9.16"}}
            (root / "config/environment.json").write_text(json.dumps(manifest))
            with patch.object(installer, "ROOT", root), patch.object(installer.shutil, "disk_usage", return_value=type("Disk", (), {"free": 100 * 1024**3})()):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    installer.install()
            self.assertFalse((root / "carla/CARLA_0.9.16").exists())

    def test_completed_install_reuses_archive_even_when_disk_free_is_low(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            package_root = root / "carla/CARLA_0.9.16"
            package_root.mkdir(parents=True)
            for name in installer.REQUIRED_FILES:
                path = package_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test executable")
            archive = root / "carla/CARLA_0.9.16.zip"
            archive.write_bytes(b"previously validated ZIP")
            digest = installer.digest_file(archive)
            manifest = {"carla_version": "0.9.16", "windows_package": {
                "url": "https://example.invalid/package", "archive_bytes": archive.stat().st_size,
                "sha256": digest, "install_directory": "carla/CARLA_0.9.16"}}
            (root / "config/environment.json").write_text(json.dumps(manifest))
            (package_root / ".vlf-install.json").write_text(json.dumps({
                "archive_sha256": digest, "zip_members": 100, "uncompressed_bytes": 2000}))
            with patch.object(installer, "ROOT", root), patch.object(installer.shutil, "disk_usage", return_value=type("Disk", (), {"free": 1})()):
                report = installer.install()
            self.assertTrue(report["extraction_complete"])
            self.assertTrue(report["checksum_matched_manifest"])
            self.assertEqual(report["zip_members"], 100)

    def test_completed_marker_cannot_hide_missing_shipping_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            package_root = root / "carla/CARLA_0.9.16"
            package_root.mkdir(parents=True)
            (package_root / "CarlaUE4.exe").write_bytes(b"launcher only")
            archive = root / "carla/CARLA_0.9.16.zip"
            archive.write_bytes(b"previously validated ZIP")
            digest = installer.digest_file(archive)
            (root / "config/environment.json").write_text(json.dumps({"carla_version": "0.9.16", "windows_package": {
                "url": "https://example.invalid/package", "archive_bytes": archive.stat().st_size,
                "sha256": digest, "install_directory": "carla/CARLA_0.9.16"}}))
            (package_root / ".vlf-install.json").write_text(json.dumps({"archive_sha256": digest}))
            with patch.object(installer, "ROOT", root), self.assertRaisesRegex(ValueError, "marker or executable differs"):
                installer.install()


if __name__ == "__main__":
    unittest.main()
