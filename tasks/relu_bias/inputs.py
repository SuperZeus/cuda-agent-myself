import torch


def get_init_inputs():
    return [torch.linspace(-0.5, 0.5, 64)]


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(200 + seed)
    return [torch.randn(16, 64, generator=g)]
