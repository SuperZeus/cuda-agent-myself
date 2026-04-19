import torch


def get_init_inputs():
    return []


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(700 + seed)
    return [torch.randn(16, 32, 64, generator=g)]
