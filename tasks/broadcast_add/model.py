import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        return x + self.bias
