"""Protocol-fixed bearing-level split definitions."""
from pathlib import Path
import pandas as pd

INTRA = {
 "train": ["Bearing1_1","Bearing1_2","Bearing1_4","Bearing2_1","Bearing2_2","Bearing2_4","Bearing3_1","Bearing3_2","Bearing3_4"],
 "validation": ["Bearing1_3","Bearing2_3","Bearing3_3"],
 "test": ["Bearing1_5","Bearing2_5","Bearing3_5"],
}

def _check(groups):
    all_ids = [x for values in groups.values() for x in values]
    if len(all_ids) != len(set(all_ids)): raise ValueError("bearing overlap across split groups")
    return groups

def loco_splits():
    # Each split holds out one condition for testing and uses one bearing from
    # each remaining condition for validation; all other bearings are training.
    result = {}
    for held in (1, 2, 3):
        test = [f"Bearing{held}_{i}" for i in range(1, 6)]
        validation = [f"Bearing{c}_3" for c in (1, 2, 3) if c != held]
        excluded = set(test) | set(validation)
        train = [f"Bearing{c}_{i}" for c in (1, 2, 3) for i in range(1, 6) if f"Bearing{c}_{i}" not in excluded]
        result[f"loco_{held}"] = _check({"train": train, "validation": validation, "test": test})
    return result

def write_splits(path="data/processed/split.csv"):
    rows=[]
    for split, groups in [("intra", INTRA), *loco_splits().items()]:
        for subset, bearings in groups.items():
            rows.extend({"split": split, "subset": subset, "bearing": b} for b in bearings)
    out=Path(path); out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); return pd.DataFrame(rows)

def load_split(split="intra", path="data/processed/split.csv"):
    frame=pd.read_csv(path); frame=frame[frame.split == split]
    if frame.empty: raise ValueError(f"unknown split: {split}")
    return {k: frame.loc[frame.subset==k,"bearing"].tolist() for k in ("train","validation","test")}

if __name__ == "__main__": write_splits()

# Compatibility interface retained for synthetic pipeline tests.
def split_bearings(bearings, train, validation, test):
    groups={"train":list(train),"validation":list(validation),"test":list(test)}
    _check(groups)
    unknown=set(sum(groups.values(),[]))-set(bearings)
    if unknown: raise ValueError(f"unknown bearings: {sorted(unknown)}")
    return groups
