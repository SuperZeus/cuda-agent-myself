import torch


def get_init_inputs():
    return [torch.tensor(1e-4)]


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(900 + seed)
    return [torch.randn(16, 64, generator=g)]
