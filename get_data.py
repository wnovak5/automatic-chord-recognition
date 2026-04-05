from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


RECORD_ID = "5794629"
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
BASE_DOWNLOAD_URL = f"{RECORD_URL}/files"
DEFAULT_ROOT = Path("data/raw/AAM")
DEFAULT_ARCHIVE_DIRNAME = "_archives"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    md5: str
    category: str

    @property
    def url(self) -> str:
        return f"{BASE_DOWNLOAD_URL}/{self.filename}?download=1"


ARCHIVES: tuple[ArchiveSpec, ...] = (
    ArchiveSpec("0001-1000-annotations-v1.1.0.zip", "fab44c247f53292e7f4c37d4850aed70", "annotations"),
    ArchiveSpec("0001-1000-audio-mixes.zip", "55feacbd10405191098f1906174c374b", "audio-mixes-mp3"),
    ArchiveSpec("0001-1000-audio-multitracks.zip", "84205a58efb75218ecf983452eebd6c5", "audio-multitracks-mp3"),
    ArchiveSpec("0001-1000-midis.zip", "ced1350c5324febc79dec467f0296008", "midis"),
    ArchiveSpec("1001-2000-annotations-v1.1.0.zip", "5ad193fc1bbc5987436667371771eecb", "annotations"),
    ArchiveSpec("1001-2000-audio-mixes.zip", "35958837ad306a5cf064c775670d7efe", "audio-mixes-mp3"),
    ArchiveSpec("1001-2000-audio-multitracks.zip", "a9ed6b1fb0badc412682f5117c41950c", "audio-multitracks-mp3"),
    ArchiveSpec("1001-2000-midis.zip", "98ce17c271aa3faf72b849ebd602198e", "midis"),
    ArchiveSpec("2001-3000-annotations-v1.1.0.zip", "61956732ad1751f9491a3ba4983a763e", "annotations"),
    ArchiveSpec("2001-3000-audio-mixes.zip", "cf716165098919cfa1b144705204ebfc", "audio-mixes-mp3"),
    ArchiveSpec("2001-3000-audio-multitracks.zip", "6f3e9059c93fffa3dd56decd000508b5", "audio-multitracks-mp3"),
    ArchiveSpec("2001-3000-midis.zip", "a6d40ba074eac2b9e72f51d0203de2b3", "midis"),
    ArchiveSpec("info.zip", "15b395250d43a11e899908319e750b2f", "info"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the full AAM dataset from Zenodo and extract it into the "
            "directory layout expected by this repository."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Destination dataset root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--archives-dir",
        type=Path,
        default=None,
        help="Directory to store the downloaded zip archives. Defaults to <root>/_archives.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify the archives without extracting them.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract existing archives without downloading them.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded archives after successful extraction.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification for local archives.",
    )
    args = parser.parse_args()

    if args.download_only and args.extract_only:
        parser.error("--download-only and --extract-only cannot be used together")

    return args


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def print_progress(prefix: str, downloaded: int, total: int | None) -> None:
    if total and total > 0:
        percent = (downloaded / total) * 100
        message = f"\r{prefix}: {format_size(downloaded)} / {format_size(total)} ({percent:5.1f}%)"
    else:
        message = f"\r{prefix}: {format_size(downloaded)}"
    print(message, end="", flush=True)


def download_archive(spec: ArchiveSpec, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = archive_path.stat().st_size if archive_path.exists() else 0
    headers: dict[str, str] = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    request = urllib.request.Request(spec.url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            supports_resume = response.status == 206 and existing_size > 0
            if existing_size > 0 and not supports_resume:
                archive_path.unlink()
                existing_size = 0

            mode = "ab" if supports_resume else "wb"
            response_length = _response_length(response)
            total_size = existing_size + response_length if response_length is not None else None
            downloaded = existing_size

            with archive_path.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    print_progress(f"Downloading {spec.filename}", downloaded, total_size)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download {spec.filename}: {exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {spec.filename}: {exc}") from exc

    print()


def _response_length(response: urllib.response.addinfourl) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total) - int(content_range.split()[1].split("-", 1)[0])

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return int(content_length)
    return None


def destination_for_member(root: Path, spec: ArchiveSpec, member_name: str) -> Path:
    member_path = Path(member_name)
    if spec.category == "info":
        cleaned_parts = [part for part in member_path.parts if part not in ("", ".", "..")]
        return root / "info" / Path(*cleaned_parts)
    return root / spec.category / member_path.name


def extract_archive(spec: ArchiveSpec, archive_path: Path, root: Path) -> tuple[int, int]:
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        total = len(members)
        for index, member in enumerate(members, start=1):
            destination = destination_for_member(root, spec, member.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists() and destination.stat().st_size == member.file_size:
                skipped += 1
                print(
                    f"\rExtracting {spec.filename}: {index}/{total} files "
                    f"(extracted={extracted}, skipped={skipped})",
                    end="",
                    flush=True,
                )
                continue

            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
            extracted += 1
            print(
                f"\rExtracting {spec.filename}: {index}/{total} files "
                f"(extracted={extracted}, skipped={skipped})",
                end="",
                flush=True,
            )

    print()
    return extracted, skipped


def verify_archive(spec: ArchiveSpec, archive_path: Path, skip_md5: bool) -> None:
    if skip_md5:
        return

    actual_md5 = compute_md5(archive_path)
    if actual_md5 != spec.md5:
        raise RuntimeError(
            f"MD5 mismatch for {archive_path.name}: expected {spec.md5}, got {actual_md5}"
        )


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    archives_dir = (
        args.archives_dir.expanduser().resolve()
        if args.archives_dir is not None
        else root / DEFAULT_ARCHIVE_DIRNAME
    )

    print(f"Zenodo record: {RECORD_URL}")
    print(f"Dataset root: {root}")
    print(f"Archive cache: {archives_dir}")

    if not args.extract_only:
        archives_dir.mkdir(parents=True, exist_ok=True)

    for spec in ARCHIVES:
        archive_path = archives_dir / spec.filename

        if not args.extract_only:
            archive_ready = False
            if archive_path.exists():
                print(f"Found existing archive {archive_path.name}")
                if args.skip_md5:
                    archive_ready = True
                else:
                    actual_md5 = compute_md5(archive_path)
                    if actual_md5 == spec.md5:
                        archive_ready = True
                    else:
                        print(
                            f"Archive checksum mismatch for {archive_path.name}; "
                            "attempting to resume or re-download it."
                        )

            if not archive_ready:
                download_archive(spec, archive_path)
            verify_archive(spec, archive_path, skip_md5=args.skip_md5)

        if args.download_only:
            continue

        if not archive_path.exists():
            raise FileNotFoundError(
                f"Archive not found for extraction: {archive_path}. "
                "Run without --extract-only or point --archives-dir at the downloaded zips."
            )

        verify_archive(spec, archive_path, skip_md5=args.skip_md5)
        extracted, skipped = extract_archive(spec, archive_path, root)
        print(f"{spec.filename}: extracted {extracted} files, skipped {skipped}")

    if not args.download_only and not args.keep_archives and archives_dir.exists():
        shutil.rmtree(archives_dir)
        print(f"Removed archive cache {archives_dir}")

    print("AAM dataset is ready.")
    print(f"Your existing loaders will now prefer {root} automatically when it exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
