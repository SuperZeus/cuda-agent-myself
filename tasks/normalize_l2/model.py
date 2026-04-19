import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps: torch.Tensor):
        super().__init__()
        self.eps = nn.Parameter(eps.clone())

    def forward(self, x):
        denom = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + self.eps)
        return x / denom
