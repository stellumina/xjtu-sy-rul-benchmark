import math
from src.metrics import mae, phm08_score, rmse


def test_rmse_known_answer():
    assert abs(rmse([1, 2], [2, 2]) - math.sqrt(0.5)) < 1e-12


def test_mae_known_answer():
    assert mae([1, 2, 4], [2, 1, 1]) == 5 / 3


def test_phm08_asymmetric_penalty_and_filter():
    # Errors are +10 (overprediction) and -13 (underprediction), each score e-1.
    expected = 2 * (math.e - 1)
    assert abs(phm08_score([0, 0], [10, -13]) - expected) < 1e-12
    # The RUL=5 sample is excluded by the protocol truncation filter.
    assert abs(mae([0, 100], [10, 0], rul_true=[2, 5], truncation=2) - 10.0) < 1e-12
