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

    def __init__(self, in_features: int=256, bottom_channels: int=4, pyramid_depth: int=4, step: int=2):
        super().__init__()

        self.in_features = in_features
        self.bottom_channels = bottom_channels
        self.step = step

        self.layers = nn.ModuleList([])
        sizes = []

        previous = self.bottom_channels
        for i in range(pyramid_depth):
            if i < 1:
                sizes.append((previous, previous))
            else:
                if i % 2 == 0:
                    sizes.append((previous * 2, previous * 2))
                    previous = sizes[i][0]

                else:
                    sizes.append((
                        (previous * 2) - 1,
                        (previous * 2) - 1
                    ))

                    previous = sizes[i][0]

            self.layers.append(nn.Conv2d(self.in_features, self.in_features,
                                         kernel_size=(sizes[i][0] - 2,
                                                      sizes[i][1] - 2)))

    def forward(self, pyramid_planes: List[torch.Tensor]) -> torch.Tensor:
        for i, (plane, conv_reducer) in enumerate(zip(pyramid_planes, self.layers)):
            pyramid_planes[i] = conv_reducer(plane)

        return torch.stack(pyramid_planes)
