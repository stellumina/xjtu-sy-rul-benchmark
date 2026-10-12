"""Metric interfaces."""
def mean_absolute_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
