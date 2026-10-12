from pathlib import Path
import pandas as pd
from src.data import read_bearing
from src.label import rul_labels
from src.split import split_bearings

def test_synthetic_pipeline(tmp_path: Path):
    root = tmp_path / "raw"
    for bearing in ("Bearing1_1", "Bearing2_1"):
        folder = root / "35Hz12kN" / bearing
        folder.mkdir(parents=True)
        for i in range(1, 6):
            pd.DataFrame({"vibration": [i, i + 1]}).to_csv(folder / f"{i}.csv", index=False)
    frames = read_bearing(root, "Bearing1_1")
    assert len(frames) == 5
    assert frames[0]["vibration"].tolist() == [1, 2]
    groups = split_bearings(["Bearing1_1", "Bearing2_1"], ["Bearing1_1"], [], ["Bearing2_1"])
    assert groups["test"] == ["Bearing2_1"]
    assert rul_labels(5) == [4, 3, 2, 1, 0]
