import torch
import torch.nn as nn

try:
    import cuda_extension
except Exception:
    cuda_extension = None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if cuda_extension is not None and x.is_cuda:
            return cuda_extension.mean_lastdim_forward(x.contiguous(), 0)
        return x.mean(dim=-1)
