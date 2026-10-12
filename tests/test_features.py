import math
import numpy as np
from scripts.extract_features import compute_features


def test_known_sine_and_pulse_features():
    # Horizontal: [1, -1, 1, -1] has RMS=1 and excess kurtosis=-2.
    # Vertical: a unit pulse has RMS=1/2, peak=1, and crest=2.
    signal = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
    result = compute_features(signal)
    assert abs(result["rms_h"] - math.sqrt(1.0)) < 1e-6
    assert abs(result["kurt_h"] - (-2.0)) < 1e-6
    assert abs(result["crest_h"] - 1.0) < 1e-6
    assert abs(result["rms_v"] - 0.5) < 1e-6
    assert abs(result["peak_v"] - 1.0) < 1e-6
    assert abs(result["crest_v"] - 2.0) < 1e-6
