import torch


def get_init_inputs():
    return [torch.linspace(-1.0, 1.0, 128)]


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(800 + seed)
    return [torch.randn(32, 128, generator=g)]
