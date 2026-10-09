"""Atomic metadata replacement and exclusive publication under I/O denial."""

from __future__ import annotations

import errno
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import call, patch

from carla_tools import episode
from scripts.carla_rgb_probe import png_from_rgb
from test_episode import fixture, record


def permission_denied(winerror):
    error = PermissionError(errno.EACCES, "injected-access-denial")
    error.winerror = winerror
    return error


class EpisodeWindowsIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.real_replace = episode.os.replace
        self.real_sleep = time.sleep

    def test_transient_win5_replaces_complete_bytes_using_same_pending_file(self):
        for denial_count in (1, 2):
            with self.subTest(denial_count=denial_count):
                destination = self.root / f"metadata-{denial_count}.json"
                old_bytes, new_bytes = b'{"status":"RUNNING"}\n', b'{"status":"COMPLETE"}\n'
                destination.write_bytes(old_bytes)
                sources = []

                def deny_then_replace(source, target):
                    source, target = Path(source), Path(target)
                    sources.append(source)
                    self.assertEqual(source.suffix, ".pending")
                    self.assertEqual(source.read_bytes(), new_bytes)
                    self.assertEqual(target.read_bytes(), old_bytes)
                    if len(sources) <= denial_count:
                        raise permission_denied(5)
                    return self.real_replace(source, target)

                with patch("carla_tools.episode.os.replace", side_effect=deny_then_replace) as replace, \
                        patch("carla_tools.episode.time.sleep", wraps=self.real_sleep) as sleep:
                    episode._atomic_bytes(destination, new_bytes, exclusive=False)
                self.assertEqual(replace.call_count, denial_count + 1)
                self.assertEqual(sleep.call_args_list, [call(0.01 * (index + 1)) for index in range(denial_count)])
                self.assertEqual(len(set(sources)), 1)
                self.assertEqual(destination.read_bytes(), new_bytes)
                self.assertFalse(sources[0].exists())
                self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_permanent_win5_is_bounded_and_preserves_old_destination(self):
        destination = self.root / "metadata.json"
        old_bytes, new_bytes = b"old-metadata", b"new-complete-metadata"
        destination.write_bytes(old_bytes)
        sources = []
        error = permission_denied(5)

        def deny(source, target):
            sources.append(Path(source))
            self.assertEqual(Path(source).read_bytes(), new_bytes)
            self.assertEqual(Path(target).read_bytes(), old_bytes)
            raise error

        with patch("carla_tools.episode.os.replace", side_effect=deny) as replace, \
                patch("carla_tools.episode.time.sleep", wraps=self.real_sleep) as sleep:
            with self.assertRaises(PermissionError) as raised:
                episode._atomic_bytes(destination, new_bytes, exclusive=False)
        self.assertIs(raised.exception, error)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.01), call(0.02)])
        self.assertEqual(len(set(sources)), 1)
        self.assertEqual(destination.read_bytes(), old_bytes)
        self.assertFalse(sources[0].exists())
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_posix_permission_denial_is_not_retried(self):
        destination = self.root / "metadata.json"
        destination.write_bytes(b"old")
        error = permission_denied(None)
        with patch("carla_tools.episode.os.replace", side_effect=error) as replace, \
                patch("carla_tools.episode.time.sleep") as sleep:
            with self.assertRaises(PermissionError) as raised:
                episode._atomic_bytes(destination, b"new", exclusive=False)
        self.assertIs(raised.exception, error)
        self.assertEqual(raised.exception.errno, errno.EACCES)
        self.assertIsNone(raised.exception.winerror)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_other_windows_permission_code_is_not_retried(self):
        destination = self.root / "metadata.json"
        destination.write_bytes(b"old")
        error = permission_denied(32)
        with patch("carla_tools.episode.os.replace", side_effect=error) as replace, \
                patch("carla_tools.episode.time.sleep") as sleep:
            with self.assertRaises(PermissionError) as raised:
                episode._atomic_bytes(destination, b"new", exclusive=False)
        self.assertIs(raised.exception, error)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_other_os_error_propagates_without_retry(self):
        for error_number in (errno.EIO, errno.ENOSPC):
            with self.subTest(errno=error_number):
                destination = self.root / f"metadata-{error_number}.json"
                destination.write_bytes(b"old")
                error = OSError(error_number, "injected-io-failure")
                with patch("carla_tools.episode.os.replace", side_effect=error) as replace, \
                        patch("carla_tools.episode.time.sleep") as sleep:
                    with self.assertRaises(OSError) as raised:
                        episode._atomic_bytes(destination, b"new", exclusive=False)
                self.assertIs(raised.exception, error)
                replace.assert_called_once()
                sleep.assert_not_called()
                self.assertEqual(destination.read_bytes(), b"old")
                self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_exclusive_existing_png_is_never_replaced_or_retried(self):
        destination = self.root / "existing.png"
        old_bytes = png_from_rgb(bytes([1, 2, 3] * 16 * 16), 16, 16)
        new_bytes = png_from_rgb(bytes([4, 5, 6] * 16 * 16), 16, 16)
        destination.write_bytes(old_bytes)
        real_link = episode.os.link
        with patch("carla_tools.episode.os.link", wraps=real_link) as link, \
                patch("carla_tools.episode.os.replace") as replace, \
                patch("carla_tools.episode.time.sleep") as sleep:
            with self.assertRaises(FileExistsError):
                episode._atomic_bytes(destination, new_bytes, exclusive=True)
        link.assert_called_once()
        replace.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(destination.read_bytes(), old_bytes)
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_exclusive_win5_denial_has_no_retry_or_partial_destination(self):
        destination = self.root / "new.png"
        error = permission_denied(5)
        with patch("carla_tools.episode.os.link", side_effect=error) as link, \
                patch("carla_tools.episode.os.replace") as replace, \
                patch("carla_tools.episode.time.sleep") as sleep:
            with self.assertRaises(PermissionError) as raised:
                episode._atomic_bytes(destination, b"complete-image-bytes", exclusive=True)
        self.assertIs(raised.exception, error)
        link.assert_called_once()
        replace.assert_not_called()
        sleep.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob("*.pending")), [])

    def test_finalize_permanent_denial_never_persists_complete_metadata(self):
        _, metadata, cleanup, png = fixture()
        destination = self.root / "episode"
        writer = episode.EpisodeWriter(destination, metadata)
        writer.append(record(100), png)
        writer.append(record(101), png)
        metadata_path = destination / "episode.json"
        original = metadata_path.read_bytes()
        self.assertEqual(json.loads(original)["status"], "RUNNING")
        attempts = []

        def deny(source, target):
            pending = Path(source)
            attempts.append((pending, json.loads(pending.read_bytes())["status"]))
            self.assertEqual(Path(target), metadata_path)
            self.assertEqual(metadata_path.read_bytes(), original)
            raise permission_denied(5)

        with patch("carla_tools.episode.os.replace", side_effect=deny) as replace, \
                patch("carla_tools.episode.time.sleep", wraps=self.real_sleep) as sleep:
            with self.assertRaisesRegex(episode.EpisodeError, "^metadata-write-failed$"):
                writer.finalize(True, cleanup)
        # COMPLETE publication and the best-effort FAILED publication are each
        # bounded independently, with a fresh complete pending file per write.
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(sleep.call_args_list, [call(0.01), call(0.02), call(0.01), call(0.02)])
        self.assertEqual([status for _, status in attempts], ["COMPLETE"] * 3 + ["FAILED"] * 3)
        self.assertEqual(len({source for source, _ in attempts[:3]}), 1)
        self.assertEqual(len({source for source, _ in attempts[3:]}), 1)
        self.assertNotEqual(attempts[0][0], attempts[3][0])
        self.assertEqual(metadata_path.read_bytes(), original)
        self.assertEqual(writer.metadata["status"], "FAILED")
        self.assertEqual(writer.metadata["error"], {"code": "metadata-write-failed"})
        self.assertNotIn("injected-access-denial", json.dumps(writer.metadata))
        self.assertEqual(list(destination.rglob("*.pending")), [])
        validation = episode.validate_episode(destination)
        self.assertEqual(validation["status"], "FAIL")
        self.assertFalse(validation["complete"])
        self.assertIn("episode-incomplete", {item["code"] for item in validation["errors"]})


if __name__ == "__main__":
    unittest.main()
