import torch


def get_init_inputs():
    return [torch.tensor(0.15), torch.tensor(1.25)]


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(500 + seed)
    return [torch.randn(32, 64, generator=g)]
