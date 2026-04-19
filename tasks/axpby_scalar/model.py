import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.alpha = nn.Parameter(alpha.clone())
        self.beta = nn.Parameter(beta.clone())

    def forward(self, x, y):
        return self.alpha * x + self.beta * y
