"""Create a compact label sanity report from feature CSVs."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Allow direct execution from the repository root or any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.label import diagnose_bearing, labels_for_bearing


def synthetic_features(condition: int, index: int) -> pd.DataFrame:
    n_files = 10
    # The second file crosses the 3-sigma threshold computed from file one.
    rms = [1.0, 2.0] + [2.0 + i * 0.1 for i in range(n_files - 2)]
    return pd.DataFrame({"file_idx": range(1, n_files + 1), "rms_h": rms, "rms_v": rms})


def main() -> int:
    feature_dir = Path("data/features")
    rows: list[dict[str, int | str]] = []
    real_files = sorted(p for p in feature_dir.glob("Bearing*_*.csv") if p.name != "labels.csv")
    synthetic = not real_files
    sources = []
    if real_files:
        sources = [(p.stem, pd.read_csv(p)) for p in real_files]
    else:
        sources = [(f"Bearing{condition}_{index}", synthetic_features(condition, index)) for condition in range(1, 4) for index in range(1, 6)]
    for bearing, frame in sources:
        condition = int(bearing.removeprefix("Bearing").split("_")[0])
        total, start, cap = diagnose_bearing(frame)
        rows.append({"bearing": bearing, "condition": condition, "寿命文件数": total, "退化起点": start, "RUL_start": cap})
    report = Path("reports/label_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Label sanity report\n\n")
        if synthetic:
            handle.write("**synthetic**：data/features/ 尚无真实特征表，本报告使用合成特征生成。\n\n")
        else:
            handle.write("使用 data/features/ 中的特征表生成。\n\n")
        handle.write("| bearing | condition | 寿命文件数 | 退化起点 | RUL_start |\n|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(f"| {row['bearing']} | {row['condition']} | {row['寿命文件数']} | {row['退化起点']} | {row['RUL_start']} |\n")
    print(f"wrote {report} ({'synthetic' if synthetic else 'real'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
