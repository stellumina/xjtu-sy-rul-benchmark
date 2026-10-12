import pandas as pd
from src.label import diagnose_bearing, labels_for_bearing


def test_three_sigma_start_and_truncated_rul():
    # 10 files: first file is the 10% healthy segment; file 4 is first crossing.
    features = pd.DataFrame({"file_idx": range(1, 11), "rms_h": [1.0, 1.0, 1.0, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2], "rms_v": [0.5] * 10})
    labels, diagnosis = labels_for_bearing(features)
    assert diagnosis == [10, 4, 6]
    assert labels["rul"].tolist() == [6, 6, 6, 6, 5, 4, 3, 2, 1, 0]
    assert labels["rul"].is_monotonic_decreasing
    assert labels.iloc[-1]["rul"] == 0
    assert not bool(labels.iloc[0]["is_degraded"])


def test_diagnosis_interface():
    features = pd.DataFrame({"file_idx": [1, 2], "rms_h": [1.0, 1.0], "rms_v": [1.0, 1.0]})
    assert diagnose_bearing(features) == [2, 2, 0]
