"""Leakage-safe bearing-level splits."""
def split_bearings(bearings, train, validation, test):
    groups = {"train": list(train), "validation": list(validation), "test": list(test)}
    assigned = [x for values in groups.values() for x in values]
    if len(assigned) != len(set(assigned)):
        raise ValueError("a bearing appears in more than one split")
    unknown = set(assigned) - set(bearings)
    if unknown:
        raise ValueError(f"unknown bearings: {sorted(unknown)}")
    return groups
