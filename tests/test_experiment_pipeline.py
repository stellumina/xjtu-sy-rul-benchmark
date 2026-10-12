from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import FeatureLoader, FEATURE_COLUMNS
from src.label import labels_for_bearing
from src.models.baselines import MLP
from src.split import INTRA, loco_splits, split_bearings
from src.train import normalize, seed_all

def frame(n=8, offset=0):
    d={"file_idx":range(1,n+1)}
    for i,c in enumerate(FEATURE_COLUMNS): d[c]=np.linspace(1+offset+i,2+offset+i,n)
    return pd.DataFrame(d)

def test_loader_windows_and_mask(tmp_path):
    (tmp_path/"Bearing1_1.csv").write_text(frame(2).to_csv(index=False))
    loader=FeatureLoader(tmp_path); windows,masks,meta=loader.make_windows(3)
    assert windows.shape==(2,3,8); assert masks[0].tolist()==[0,0,1]; assert np.all(windows[0,0]==windows[0,1])

def test_split_disjoint_and_normalization_train_only():
    groups=split_bearings(sum(INTRA.values(),[]),INTRA["train"],INTRA["validation"],INTRA["test"])
    assert not (set(groups["train"]) & set(groups["test"]))
    train=np.array([[0.],[2.]]); test=np.array([[100.]])
    normalized, test_norm, mean, std=normalize(train,test)
    assert mean[0]==1 and test_norm[0,0]>90
    assert set(loco_splits())=={"loco_1","loco_2","loco_3"}

def test_small_model_same_seed_is_deterministic():
    x=torch.ones(4,8); y=torch.arange(4,dtype=torch.float32)
    def run():
        seed_all(7); net=MLP(8); opt=torch.optim.SGD(net.parameters(),lr=.01)
        for _ in range(2): opt.zero_grad(); loss=((net(x)-y)**2).mean(); loss.backward(); opt.step()
        return net(x).detach().numpy()
    np.testing.assert_array_equal(run(),run())
