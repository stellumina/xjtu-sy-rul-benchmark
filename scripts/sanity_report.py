"""Generate real-data label diagnostics and RMS figures."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.label import diagnose_bearing

def main():
    files=sorted(Path("data/features").glob("Bearing*_*.csv"))
    if not files:
        print("No real feature data found in data/features; sanity report skipped."); return 0
    import matplotlib.pyplot as plt
    out=Path("reports"); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in files:
        frame=pd.read_csv(path); n,start,cap=diagnose_bearing(frame); condition=path.stem.removeprefix("Bearing").split("_")[0]
        rows.append((path.stem,condition,n,start,100*start/n,cap))
        plt.figure(); plt.plot(frame.file_idx,frame.rms_h,label="rms_h"); plt.plot(frame.file_idx,frame.rms_v,label="rms_v"); plt.axvline(start,color="red"); plt.legend(); plt.xlabel("file_idx"); plt.ylabel("RMS"); plt.savefig(figdir/f"{path.stem}.png"); plt.close()
    with (out/"label_report.md").open("w",encoding="utf8") as f:
        f.write("# Label report\n\n| bearing | condition | n_files | 3σ起点 file_idx | 起点占寿命百分比 | 截断RUL | 异常 |\n|---|---:|---:|---:|---:|---:|---|\n")
        for b,c,n,s,p,cap in rows: f.write(f"| {b} | {c} | {n} | {s} | {p:.2f}% | {cap} |  |\n")
    print(f"wrote reports/label_report.md and {len(rows)} figures"); return 0
if __name__=="__main__": raise SystemExit(main())
