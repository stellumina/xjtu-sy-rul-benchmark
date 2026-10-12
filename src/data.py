"""Data loading helpers for XJTU-SY CSV files."""
from pathlib import Path
import pandas as pd

def list_bearing_files(root: str | Path, bearing: str | None = None) -> list[Path]:
    base = Path(root)
    files = sorted(base.glob(f"**/{bearing}/*.csv")) if bearing else sorted(base.glob("**/Bearing*/*.csv"))
    return files

def read_bearing(root: str | Path, bearing: str) -> list[pd.DataFrame]:
    return [pd.read_csv(path) for path in list_bearing_files(root, bearing)]
