"""Run all baseline/seed/split combinations with resume support."""
from pathlib import Path
import subprocess, sys
import pandas as pd

def main():
    files=list(Path("data/features").glob("Bearing*_*.csv"))
    if not files:
        print("No real feature data found in data/features; generate and push features first."); return 0
    done=set()
    if Path("results/results.csv").exists():
        old=pd.read_csv("results/results.csv"); done=set(zip(old.model.astype(str),old.seed.astype(int),old.split.astype(str)))
    for model in ("mlp","cnn","gradient_boosting"):
        for seed in range(5):
            for split in ("intra","loco_1"):
                if (model,seed,split) in done: continue
                subprocess.run([sys.executable,"-m","src.train","--model",model,"--seed",str(seed),"--split",split],check=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
