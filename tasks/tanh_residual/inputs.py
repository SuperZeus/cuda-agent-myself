import torch


def get_init_inputs():
    return [torch.tensor(0.25)]


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(400 + seed)
    return [torch.randn(8, 128, generator=g)]
