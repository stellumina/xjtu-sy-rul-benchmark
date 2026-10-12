"""Protocol-compliant RUL label construction from feature tables."""
from __future__ import annotations

import math
from pathlib import Path
import re
import pandas as pd

FEATURE_COLUMNS = ["rms_h", "rms_v"]
LABEL_COLUMNS = ["condition", "bearing", "file_idx", "rul", "is_degraded"]


def _rms_series(features: pd.DataFrame) -> pd.Series:
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"missing RMS columns: {missing}")
    return features[FEATURE_COLUMNS].max(axis=1)


def labels_for_bearing(features: pd.DataFrame) -> tuple[pd.DataFrame, list[int | float]]:
    """Return labels and diagnostics for one bearing, using only its own rows.

    The input must be in file order or contain a numeric ``file_idx`` column.
    Healthy statistics use ceil(10% of this bearing's files), with at least one
    file. A missing threshold crossing is conservatively represented by the last
    file as the degradation start.
    """
    if features.empty:
        raise ValueError("cannot label an empty bearing")
    frame = features.copy()
    if "file_idx" in frame:
        frame = frame.sort_values("file_idx", kind="stable").reset_index(drop=True)
    n_files = len(frame)
    healthy_count = max(1, math.ceil(n_files * 0.10))
    rms = _rms_series(frame)
    healthy = rms.iloc[:healthy_count]
    threshold = float(healthy.mean() + 3.0 * healthy.std(ddof=0))
    crossing = rms.gt(threshold)
    start_position = int(crossing.idxmax()) if crossing.any() else n_files - 1
    start_file = int(frame.iloc[start_position]["file_idx"]) if "file_idx" in frame else start_position + 1
    cap = n_files - start_position - 1
    natural_rul = pd.Series(range(n_files - 1, -1, -1), index=frame.index, dtype=int)
    labels = natural_rul.clip(upper=cap)
    result = pd.DataFrame({
        "file_idx": frame["file_idx"].astype(int).to_numpy() if "file_idx" in frame else range(1, n_files + 1),
        "rul": labels.to_numpy(dtype=int),
        "is_degraded": (frame.index.to_numpy() >= start_position),
    })
    return result, [n_files, start_file, cap]


def diagnose_bearing(features: pd.DataFrame) -> list[int | float]:
    """Return ``[n_files, degradation_start_file, start_rul]``."""
    _, diagnosis = labels_for_bearing(features)
    return diagnosis


def _bearing_identity(path: Path, frame: pd.DataFrame) -> tuple[int, str]:
    match = re.fullmatch(r"Bearing(\d+)_(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"cannot infer condition/bearing from {path.name}")
    condition, index = match.groups()
    return int(condition), f"Bearing{condition}_{index}"


def build_labels(features_dir: str | Path = "data/features", output: str | Path = "data/features/labels.csv") -> pd.DataFrame:
    """Build labels independently per feature CSV and write the protocol schema."""
    feature_dir = Path(features_dir)
    records: list[pd.DataFrame] = []
    for path in sorted(feature_dir.glob("Bearing*_*.csv")):
        if path.name == "labels.csv":
            continue
        frame = pd.read_csv(path)
        condition, bearing = _bearing_identity(path, frame)
        labels, _ = labels_for_bearing(frame)
        labels.insert(0, "bearing", bearing)
        labels.insert(0, "condition", condition)
        records.append(labels[LABEL_COLUMNS])
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=LABEL_COLUMNS)
    result.to_csv(output, index=False)
    return result


# Backward-compatible small interface retained for callers of the initial skeleton.
def rul_labels(length: int, failure_index: int | None = None) -> list[int]:
    if length < 1:
        raise ValueError("length must be positive")
    failure = length - 1 if failure_index is None else failure_index
    if not 0 <= failure < length:
        raise ValueError("failure_index out of range")
    return [failure - i for i in range(length)]


# Descriptive alias for downstream experiment code.
construct_labels = build_labels
generate_labels = build_labels


def double_threshold_start(features: pd.DataFrame, multiplier=3.0, consecutive=2) -> int:
    """Alternative: first run of consecutive RMS threshold crossings."""
    rms = _rms_series(features).to_numpy(float)
    n = max(1, math.ceil(len(rms) * .1)); threshold = rms[:n].mean() + multiplier * rms[:n].std()
    hits = rms > threshold
    for i in range(len(hits) - consecutive + 1):
        if hits[i:i+consecutive].all(): return int(features.iloc[i].file_idx)
    return int(features.iloc[-1].file_idx)


def kurtosis_jump_start(features: pd.DataFrame, multiplier=3.0) -> int:
    """Alternative: first kurtosis jump over healthy mean plus 3 sigma."""
    cols=[c for c in ("kurt_h","kurt_v") if c in features]
    if not cols: raise ValueError("kurtosis columns required")
    k=features[cols].max(axis=1).to_numpy(float); n=max(1,math.ceil(len(k)*.1)); threshold=k[:n].mean()+multiplier*k[:n].std()
    hits=k>threshold
    return int(features.iloc[int(hits.argmax())].file_idx) if hits.any() else int(features.iloc[-1].file_idx)
