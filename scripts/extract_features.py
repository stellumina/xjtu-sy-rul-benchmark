"""Extract compact per-file features from a local XJTU-SY checkout.

This script is intended to run on the user's Windows machine. Raw waveform files
are read in place and never copied into the repository.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 32768
BEARINGS = [(condition, index) for condition in range(1, 4) for index in range(1, 6)]


def discover_raw_dir(raw_dir: str | Path | None) -> Path:
    """Use an explicit directory or probe the documented Windows candidates."""
    if raw_dir:
        path = Path(raw_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"raw directory does not exist: {path}")
        return path
    candidates = [Path("D:/RUL/XTJU"), Path("D:/RUL/XJTU")]
    for base in candidates:
        if base.is_dir():
            if "XJTU" in base.name.upper():
                return base
            matches = [p for p in base.glob("*/") if "XJTU" in p.name.upper()]
            if matches:
                return matches[0]
            return base
    raise FileNotFoundError("could not find D:/RUL/XTJU or D:/RUL/XJTU (or an XJTU subdirectory)")


def condition_for(name: str) -> int | None:
    compact = name.lower()
    if "35" in compact and "12" in compact:
        return 1
    if "37" in compact and "11" in compact:
        return 2
    if "40" in compact and "10" in compact:
        return 3
    return None


def find_condition_dirs(raw_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in [raw_dir, *raw_dir.glob("*"), *raw_dir.glob("*/*")]:
        if path.is_dir():
            condition = condition_for(path.name)
            if condition is not None and condition not in found:
                found[condition] = path
    return found


def compute_features(signal: np.ndarray) -> dict[str, float]:
    """Compute the eight protocol features for an N x 2 waveform."""
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError("waveform must have exactly two columns")
    output: dict[str, float] = {}
    for channel, suffix in enumerate(("h", "v")):
        values = np.asarray(signal[:, channel], dtype=float)
        rms = float(np.sqrt(np.mean(values * values)))
        centered = values - float(np.mean(values))
        variance = float(np.mean(centered * centered))
        kurtosis = float(np.mean(centered ** 4) / (variance ** 2) - 3.0) if variance else 0.0
        peak = float(np.max(np.abs(values)))
        output[f"rms_{suffix}"] = rms
        output[f"kurt_{suffix}"] = kurtosis
        output[f"peak_{suffix}"] = peak
        output[f"crest_{suffix}"] = peak / rms if rms else math.inf
    return output


def read_waveform(path: Path) -> np.ndarray:
    """Read comma or tab separated, headerless waveform data."""
    frame = pd.read_csv(path, sep=None, engine="python", header=None)
    values = frame.to_numpy(dtype=float)
    if values.shape != (EXPECTED_ROWS, 2):
        raise ValueError(f"expected {EXPECTED_ROWS}x2, got {values.shape[0]}x{values.shape[1] if values.ndim == 2 else 0}")
    return values


def numbered_files(folder: Path) -> list[Path]:
    files = [p for p in folder.glob("*.csv") if re.fullmatch(r"\d+", p.stem)]
    return sorted(files, key=lambda p: int(p.stem))


def extract(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bad: list[tuple[str, str]] = []
    summary: list[tuple[str, int, int]] = []
    condition_dirs = find_condition_dirs(raw_dir)
    for condition, index in BEARINGS:
        bearing = f"Bearing{condition}_{index}"
        condition_dir = condition_dirs.get(condition)
        folder = condition_dir / bearing if condition_dir else None
        rows: list[dict[str, float | int]] = []
        if folder and folder.is_dir():
            for path in numbered_files(folder):
                file_idx = int(path.stem)
                try:
                    features = compute_features(read_waveform(path))
                    rows.append({"file_idx": file_idx, **features})
                except (ValueError, OSError, UnicodeError) as exc:
                    print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
                    bad.append((str(path), str(exc)))
        else:
            print(f"WARNING: missing bearing directory: {bearing}", file=sys.stderr)
        pd.DataFrame(rows, columns=["file_idx", "rms_h", "rms_v", "kurt_h", "kurt_v", "peak_h", "peak_v", "crest_h", "crest_v"]).to_csv(output_dir / f"{bearing}.csv", index=False)
        summary.append((bearing, condition, len(rows)))
    pd.DataFrame(summary, columns=["bearing", "condition", "n_files"]).to_csv(output_dir / "meta.csv", index=False)
    with (output_dir / "bad_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "reason"])
        writer.writerows(bad)
    print("bearing condition n_files")
    for bearing, condition, count in summary:
        print(f"{bearing:>8} {condition:>9} {count:>7}")
    print(f"bad files: {len(bad)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract XJTU-SY waveform features on a local machine")
    parser.add_argument("--raw-dir", type=Path, help="raw XJTU-SY directory; defaults to D:/RUL/XTJU or D:/RUL/XJTU discovery")
    parser.add_argument("--output-dir", type=Path, default=Path("data/features"), help="feature output directory")
    args = parser.parse_args(argv)
    extract(discover_raw_dir(args.raw_dir), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
