"""Feature table loading and fixed-length temporal windows."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["rms_h","rms_v","kurt_h","kurt_v","peak_h","peak_v","crest_h","crest_v"]
ALL_COLUMNS = ["file_idx", *FEATURE_COLUMNS]

class FeatureLoader:
    def __init__(self, feature_dir="data/features"):
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.exists() or not list(self.feature_dir.glob("Bearing*_*.csv")):
            raise FileNotFoundError(f"data/features/ is empty: feature CSVs are required in {self.feature_dir}")
        self.data = {}
        self.conditions = {}
        for path in sorted(self.feature_dir.glob("Bearing*_*.csv")):
            match = re.fullmatch(r"Bearing([1-3])_([1-5])\.csv", path.name)
            if not match: continue
            frame = pd.read_csv(path)
            missing = [c for c in ALL_COLUMNS if c not in frame.columns]
            if missing: raise ValueError(f"{path} missing columns: {missing}")
            self.data[path.stem] = frame[ALL_COLUMNS].sort_values("file_idx").reset_index(drop=True)
            self.conditions[path.stem] = int(match.group(1))
        if not self.data:
            raise FileNotFoundError(f"no Bearing{{1-3}}_{{1-5}}.csv files in {self.feature_dir}")

    def load(self, bearing=None):
        return self.data[bearing] if bearing else dict(self.data)

    def make_windows(self, sequence_len=30):
        if sequence_len < 1: raise ValueError("sequence_len must be positive")
        windows, masks, metadata = [], [], []
        for bearing, frame in self.data.items():
            values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            for end in range(len(values)):
                start = max(0, end - sequence_len + 1)
                window = values[start:end + 1]
                mask = np.zeros(sequence_len, dtype=np.float32)
                mask[-len(window):] = 1.0
                padded = np.vstack([np.repeat(values[[0]], sequence_len-len(window), axis=0), window]) if len(window) < sequence_len else window
                windows.append(padded); masks.append(mask); metadata.append((bearing, int(frame.iloc[end].file_idx)))
        return np.asarray(windows), np.asarray(masks), metadata

# Compatibility helpers for the original pipeline tests.
def list_bearing_files(root, bearing=None):
    base=Path(root); return sorted(base.glob(f"**/{bearing}/*.csv")) if bearing else sorted(base.glob("**/Bearing*/*.csv"))
def read_bearing(root, bearing):
    return [pd.read_csv(path) for path in list_bearing_files(root,bearing)]
