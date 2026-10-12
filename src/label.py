"""RUL label interface."""
def rul_labels(length: int, failure_index: int | None = None):
    if length < 1:
        raise ValueError("length must be positive")
    failure = length - 1 if failure_index is None else failure_index
    if not 0 <= failure < length:
        raise ValueError("failure_index out of range")
    return [failure - i for i in range(length)]
