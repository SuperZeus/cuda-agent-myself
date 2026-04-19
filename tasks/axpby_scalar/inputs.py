import torch


def get_init_inputs():
    return [torch.tensor(1.5), torch.tensor(-0.5)]


def get_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(32, 128, generator=g)
    y = torch.randn(32, 128, generator=g)
    return [x, y]
