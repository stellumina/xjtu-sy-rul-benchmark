"""Deterministic baseline training entry point."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.data import FeatureLoader, FEATURE_COLUMNS
from src.label import build_labels
from src.metrics import compute_metrics
from src.split import load_split, write_splits
from src.models.baselines import MLP, CNN1D, GradientBoosting

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)

def _rows(loader, labels, bearings):
    x=[]; y=[]; meta=[]
    for b in bearings:
        if b not in loader.data: continue
        frame=loader.data[b]; lab=labels[labels.bearing==b].sort_values("file_idx")
        x.extend(frame[FEATURE_COLUMNS].to_numpy(float)); y.extend(lab.rul.to_numpy(float)); meta.extend((b,int(i)) for i in frame.file_idx)
    return np.asarray(x,float), np.asarray(y,float), meta

def normalize(train, *others):
    mean=train.mean(axis=0); std=train.std(axis=0); std[std==0]=1
    return ((train-mean)/std, *[((x-mean)/std) for x in others], mean, std)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["mlp","cnn","gradient_boosting"],required=True); p.add_argument("--split",default="intra"); p.add_argument("--seed",type=int,required=True); args=p.parse_args(argv)
    seed_all(args.seed); write_splits(); loader=FeatureLoader(); label_path=Path("data/features/labels.csv")
    labels=build_labels(output=label_path) if not label_path.exists() else pd.read_csv(label_path); groups=load_split(args.split)
    train_x,train_y,_=_rows(loader,labels,groups["train"]); val_x,val_y,_=_rows(loader,labels,groups["validation"]); test_x,test_y,test_meta=_rows(loader,labels,groups["test"])
    if len(train_x)==0 or len(test_x)==0: raise RuntimeError("selected split has no feature rows")
    train_x,val_x,test_x,mean,std=normalize(train_x,val_x,test_x)
    if args.model=="gradient_boosting": model=GradientBoosting(random_state=args.seed).fit(train_x,train_y); pred=model.predict(test_x)
    else:
        net=MLP(len(FEATURE_COLUMNS)) if args.model=="mlp" else CNN1D(len(FEATURE_COLUMNS)); opt=torch.optim.Adam(net.parameters(),lr=.01); tx=torch.tensor(train_x,dtype=torch.float32); ty=torch.tensor(train_y,dtype=torch.float32)
        vx=torch.tensor(val_x,dtype=torch.float32); vy=torch.tensor(val_y,dtype=torch.float32); best=None; best_loss=float("inf")
        for _ in range(2):
            opt.zero_grad(); loss=nn.functional.mse_loss(net(tx),ty); loss.backward(); opt.step()
            with torch.no_grad(): val_loss=float(nn.functional.mse_loss(net(vx),vy)) if len(vx) else float(loss)
            if val_loss < best_loss: best_loss=val_loss; best={k:v.detach().clone() for k,v in net.state_dict().items()}
        if best is not None: net.load_state_dict(best)
        with torch.no_grad(): pred=net(torch.tensor(test_x,dtype=torch.float32)).numpy()
    out=pd.DataFrame({"bearing":[m[0] for m in test_meta],"file_idx":[m[1] for m in test_meta],"true_rul":test_y,"pred_rul":pred}); Path("results/predictions").mkdir(parents=True,exist_ok=True); out.to_parquet(f"results/predictions/{args.model}_{args.seed}_{args.split}.parquet",index=False)
    metric=compute_metrics(test_y,pred,rul_true=test_y,truncation=float(np.max(test_y))); result=pd.DataFrame([["rul",args.model,args.seed,args.split,metric["rmse"],metric["mae"],metric["phm08"]]],columns=["task","model","seed","split","rmse","mae","phm08"]); Path("results").mkdir(exist_ok=True); result.to_csv("results/results.csv",mode="a",header=not Path("results/results.csv").exists(),index=False)
    return 0
if __name__=="__main__": raise SystemExit(main())
