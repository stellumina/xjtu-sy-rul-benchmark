"""Small CPU baselines used by the benchmark."""
import torch
from torch import nn
from sklearn.ensemble import GradientBoostingRegressor

class MLP(nn.Module):
    def __init__(self, input_dim=240):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(self, channels=8):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(16,1))
    def forward(self,x):
        if x.ndim == 2: x=x.unsqueeze(1)
        return self.net(x).squeeze(-1)

class GradientBoosting:
    def __init__(self, **kwargs): self.model=GradientBoostingRegressor(random_state=kwargs.pop("random_state",0), **kwargs)
    def fit(self,x,y): self.model.fit(x,y); return self
    def predict(self,x): return self.model.predict(x)
