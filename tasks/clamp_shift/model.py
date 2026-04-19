import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, offset: torch.Tensor, gain: torch.Tensor):
        super().__init__()
        self.offset = nn.Parameter(offset.clone())
        self.gain = nn.Parameter(gain.clone())

    def forward(self, x):
        return torch.clamp(x + self.offset, -1.0, 1.0) * self.gain
