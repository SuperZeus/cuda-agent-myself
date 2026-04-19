import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(scale.clone())

    def forward(self, x):
        return x + self.scale * torch.tanh(x)
