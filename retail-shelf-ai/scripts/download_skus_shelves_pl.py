#!/usr/bin/env python3
"""Download and extract SKUs on Shelves PL from Hugging Face.

Run this on MoLab, where the NVIDIA GPU is available.  The archive is large,
so keep it in persistent remote storage if the workspace is ephemeral.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REPO_ID = "shelfwise-by-form/SKUs_on_shelves_PL"
ARCHIVE_NAME = "SKUs_on_shelves_PL.zip"


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw/SKUs_on_Shelves_PL"))
    parser.add_argument("--token", default=None, help="Optional Hugging Face token; HF_TOKEN is preferred")
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install the dependency first: pip install huggingface_hub") from exc

    args.download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{ARCHIVE_NAME} …")
    archive = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=ARCHIVE_NAME,
            repo_type="dataset",
            local_dir=str(args.download_dir),
            token=args.token,
        )
    )
    print(f"Archive ready: {archive} ({archive.stat().st_size / (1024**3):.2f} GiB)")
    if args.extract_dir.exists() and any(args.extract_dir.iterdir()):
        print(f"Extraction directory already exists; skipping: {args.extract_dir}")
    else:
        print(f"Extracting to {args.extract_dir} …")
        safe_extract(archive, args.extract_dir)
        print("Extraction complete.")
    if not args.keep_archive:
        print("Archive retained because MoLab sessions can be interrupted; remove it manually after validation if storage is limited.")
    print(f"Next: python scripts/prepare_skus_shelves_pl.py --source-root {args.extract_dir}")


if __name__ == "__main__":
    main()
