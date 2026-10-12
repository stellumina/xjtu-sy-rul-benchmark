"""Download/extract XJTU-SY data without downloading by default."""
import argparse
import shutil
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

GITHUB_SOURCE = "https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets"
TARGET = Path("data/raw/XJTU-SY_Bearing_Datasets")

def parser():
    p = argparse.ArgumentParser(description="Download selected XJTU-SY bearings")
    p.add_argument("--bearings", nargs="+", metavar="BEARING", help="bearings to retain, e.g. Bearing1_1 Bearing2_3")
    p.add_argument("--url", help="confirmed archive URL; required for an actual download")
    p.add_argument("--output", type=Path, default=TARGET, help=f"extraction directory (default: {TARGET})")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if not args.url:
        print(f"No download performed. Confirm an archive URL from {GITHUB_SOURCE} first.")
        print("TODO: confirm the biaowang.tech download URL before adding it here.")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        try:
            urllib.request.urlretrieve(args.url, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp + "/extracted")
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
            raise SystemExit(f"download or extraction failed: {exc}") from exc
        source = Path(tmp) / "extracted"
        candidates = [p for p in source.rglob("*") if p.is_dir() and (not args.bearings or p.name in args.bearings)]
        for candidate in candidates:
            destination = args.output / candidate.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(candidate, destination)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
