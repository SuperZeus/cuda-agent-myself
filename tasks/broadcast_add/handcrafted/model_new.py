import torch
import torch.nn as nn

try:
    import cuda_extension
except Exception:
    cuda_extension = None


class ModelNew(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        if cuda_extension is not None and x.is_cuda:
            return cuda_extension.broadcast_add_forward(x.contiguous(), self.bias.contiguous(), 0)
        return x + self.bias
