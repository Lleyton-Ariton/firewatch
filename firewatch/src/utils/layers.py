import torch
import torch.nn

from typing import *


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x)


class PaddedConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class StackPyramidPlanes(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pyramid_planes: List[torch.Tensor]) -> torch.Tensor:
        for i, p in enumerate(pyramid_planes):
            pyramid_planes[i] = nn.Conv2d(256, 256, kernel_size=(p.size()[-2], p.size()[-1]))(p)

        return torch.stack(pyramid_planes)
