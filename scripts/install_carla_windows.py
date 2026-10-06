"""Install the pinned official Windows package into the ignored project directory.

No global Python packages, system settings, drivers, or existing installations
are changed. The ZIP is retained for repair, and extraction rejects unsafe paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024 * 1024
REQUIRED_FILES = ("CarlaUE4.exe", "CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, member: zipfile.ZipInfo) -> Path:
    name = member.filename
    parts = PurePosixPath(name).parts
    if (
        not name or not parts or "\\" in name or ":" in name or name.startswith("/")
        or ".." in parts or any(part.endswith((".", " ")) for part in parts)
        or any(PureWindowsPath(part).is_reserved() for part in parts)
        or any(ord(character) < 32 or character in '<>"|?*' for character in name)
        or stat.S_ISLNK(member.external_attr >> 16)
    ):
        raise ValueError("Unsafe ZIP member")
    destination = root.joinpath(*parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError("ZIP member escapes install directory")
    return destination


def download(url: str, archive: Path, expected_bytes: int) -> None:
    if archive.is_file():
        if archive.stat().st_size != expected_bytes:
            raise ValueError("Existing ZIP has an unexpected length; inspect it before repair")
        print("Reusing the existing ZIP.", flush=True)
        return
    partial = archive.with_suffix(".zip.part")
    if partial.exists():
        raise ValueError("A partial ZIP exists; inspect or remove that explicit file before retrying")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    received = 0
    last_progress = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200 or int(response.headers.get("Content-Length", 0)) != expected_bytes:
            raise ValueError("Official package response differs from the pinned length")
        with partial.open("xb") as stream:
            while chunk := response.read(CHUNK):
                stream.write(chunk)
                received += len(chunk)
                if received > expected_bytes:
                    raise ValueError("Download exceeded the pinned length")
                if time.monotonic() - last_progress >= 20:
                    print(f"Downloaded {received / 1024**3:.2f}/{expected_bytes / 1024**3:.2f} GiB", flush=True)
                    last_progress = time.monotonic()
    if received != expected_bytes:
        raise ValueError("Incomplete download")
    partial.rename(archive)
    print("Download complete.", flush=True)


def install(download_only: bool = False) -> dict:
    manifest = json.loads((ROOT / "config/environment.json").read_text(encoding="utf-8"))
    package = manifest["windows_package"]
    install_root = (ROOT / package["install_directory"]).resolve()
    local_root = (ROOT / "carla").resolve()
    if not local_root.is_relative_to(ROOT.resolve()) or not install_root.is_relative_to(local_root):
        raise ValueError("Install directory must remain inside the project's carla directory")
    local_root.mkdir(exist_ok=True)
    archive = local_root / f"CARLA_{manifest['carla_version']}.zip"
    expected_bytes = package["archive_bytes"]
    # Reserve archive + extraction headroom only for a new download. Existing
    # completed installs need no extra space merely to be verified and reused.
    if not archive.exists() and shutil.disk_usage(local_root).free < 35 * 1024**3:
        raise ValueError("Allow at least 35 GiB for the archive, package and extraction headroom")
    download(package["url"], archive, expected_bytes)
    sha256 = digest_file(archive)
    if package["sha256"] and sha256 != package["sha256"]:
        raise ValueError("ZIP SHA-256 mismatch; no extraction performed")
    report = {
        "schema_version": 1,
        "carla_version": manifest["carla_version"],
        "source_url": package["url"],
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256,
        "checksum_matched_manifest": bool(package["sha256"]),
        "install_directory": package["install_directory"],
        "extraction_complete": False,
    }
    if download_only:
        return report
    marker = install_root / ".vlf-install.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (previous.get("archive_sha256") != sha256
                or not all((install_root / name).is_file() for name in REQUIRED_FILES)):
            raise ValueError("Existing install marker or executable differs; inspect the installation")
        print("Verified existing installation; extraction skipped.", flush=True)
        for field in ("zip_members", "uncompressed_bytes"):
            if field in previous:
                report[field] = previous[field]
        report["extraction_complete"] = True
        return report
    if install_root.exists() and any(install_root.iterdir()):
        raise ValueError("Install directory is not empty and has no completed marker; inspect before retrying")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        targets = [(member, safe_destination(install_root, member)) for member in members]
        names = {member.filename for member in members if not member.is_dir()}
        if not all(name in names for name in REQUIRED_FILES):
            raise ValueError("Unexpected Windows package layout")
        unpacked_bytes = sum(member.file_size for member in members)
        if shutil.disk_usage(local_root).free < unpacked_bytes + 2 * 1024**3:
            raise ValueError("Insufficient space for the ZIP's uncompressed contents")
        print(f"Extracting {len(members)} members ({unpacked_bytes / 1024**3:.2f} GiB).", flush=True)
        install_root.mkdir(exist_ok=True)
        last_progress = time.monotonic()
        for index, (member, destination) in enumerate(targets, 1):
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # ZipExtFile validates each file's CRC when read to EOF.
                with source.open(member) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=CHUNK)
            if time.monotonic() - last_progress >= 20:
                print(f"Extracted {index}/{len(members)} members", flush=True)
                last_progress = time.monotonic()
    report["extraction_complete"] = True
    report["zip_members"] = len(members)
    report["uncompressed_bytes"] = unpacked_bytes
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Package extraction complete; RPC version validation remains a separate check.", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/carla_install.json")
    args = parser.parse_args()
    if os.name != "nt":
        print("This installer is for Windows. Linux/Kaggle uses its existing bootstrap.", file=sys.stderr)
        return 2
    try:
        report = install(args.download_only)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"Installation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("Installation report written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
