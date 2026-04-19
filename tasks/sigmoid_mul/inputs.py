import torch


def get_init_inputs():
    return []


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(300 + seed)
    return [torch.randn(8, 128, generator=g)]
