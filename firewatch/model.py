import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from firewatch.src.utils.layers import Flatten, PaddedConv2d, StackPyramidPlanes

from typing import *


class SmokeClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_block64 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=2),
            nn.SELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128)
        )

        self.conv_block128 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256)
        )

        self.convblock256 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512)
        )

        self.conv_block512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512)
        )

        self.dense_block = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1024, 1024),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.SELU(),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block64(x)
        x = self.conv_block128(x)
        x = self.convblock256(x)
        x = self.conv_block512(x)

        x = self.dense_block(x)

        return x


class Block(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation_function: Callable=nn.ReLU,
                 first_stride: int=False, residual: bool=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_function = activation_function
        self.first_stride = first_stride

        self.residual = residual

        if not self.first_stride:
            self.first_stride = 1

        self.block = nn.Sequential(
            PaddedConv2d(self.in_features, self.out_features, kernel_size=(3, 3), stride=self.first_stride),
            nn.InstanceNorm2d(self.out_features),
            self.activation_function(),

            PaddedConv2d(self.out_features, self.out_features, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(self.out_features),
            self.activation_function()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)

        if self.residual:
            output.add_(x)

        output = self.activation_function()(output)

        return output


class Layer(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_blocks: int=2, activation_function: Callable=nn.ReLU):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_blocks = n_blocks
        self.activation_function = activation_function

        self.layer = nn.Sequential(
            *[Block(in_features=self.in_features,
                    out_features=self.out_features,
                    activation_function=self.activation_function,
                    first_stride=2,
                    residual=False) if _ < 1 else Block(
                in_features=self.out_features,
                out_features=self.out_features,
                activation_function=self.activation_function,
                residual=True
            ) for _ in range(self.n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ResNet(nn.Module):

    RES_34_LAYER_MAP = {
        64: 3,
        128: 4,
        256: 6,
        512: 3
    }

    def __init__(self, in_features: int, layer_map: Dict[int, int]=None):
        super().__init__()

        self.in_features = in_features
        self.layer_map = layer_map
        if self.layer_map is None:
            self.layer_map = ResNet.RES_34_LAYER_MAP

        self.image_input = nn.Sequential(
            nn.Conv2d(self.in_features, 64, kernel_size=(7, 7), stride=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.layers = nn.ModuleList([])
        for i, (features, depth) in enumerate(self.layer_map.items()):
            if i > 0:
                self.layers.append(Layer(in_features=self.layers[i-1].out_features,
                                         out_features=features, n_blocks=depth))
            else:
                self.layers.append(Layer(in_features=features, out_features=features, n_blocks=depth))

        self.output_features = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),

            nn.AvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(2048, 1024),
            nn.SELU(),
            nn.Linear(1024, 1),
            nn.SELU(),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_input(x)
        for layer in self.layers:
            x = layer(x)

        x = self.output_features(x)
        x = self.fc(x)

        return x


class BottomUpPathway(ResNet):

    def __init__(self, in_features: int):
        super().__init__(in_features=in_features)
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        layer_outputs = []

        x = self.image_input(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            layer_outputs.append(x)

        return layer_outputs[::-1]


class MBlock(nn.Module):

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features

        self.block = nn.Sequential(
            nn.Conv2d(self.in_features, self.in_features, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(self.in_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        return x


class TopDownPathway(nn.Module):

    def __init__(self, top_features: int=512, in_features: int=256, n_layers: int=len(ResNet.RES_34_LAYER_MAP.keys())):
        super().__init__()

        self.top_features = top_features
        self.in_features = in_features
        self.n_layers = n_layers

        self.m_blocks, self.conv_bottlenecks = nn.ModuleList([]), nn.ModuleList([])
        for i in range(self.n_layers):
            self.m_blocks.append(
                MBlock(self.in_features),
            )

            self.conv_bottlenecks.append(nn.Conv2d(top_features, self.in_features, kernel_size=(1, 1), stride=1))
            top_features //= 2

    def forward(self, bottom_up_features: List[torch.Tensor]) -> List[torch.Tensor]:
        pyramid_feature_maps = []
        for i, (conv_bottleneck, m_block) in enumerate(zip(self.conv_bottlenecks, self.m_blocks)):
            if i < 1:
                x = conv_bottleneck(bottom_up_features[i])
                x = m_block(x)
                pyramid_feature_maps.append(x)

                x = nn.UpsamplingNearest2d(size=(bottom_up_features[i + 1].size()[-2],
                                                 bottom_up_features[i + 1].size()[-1]))(x)
            else:
                x = conv_bottleneck(bottom_up_features[i]).add_(x)
                x = m_block(x)
                pyramid_feature_maps.append(x)

                try:
                    x = nn.UpsamplingNearest2d(size=(bottom_up_features[i + 1].size()[-2],
                                                 bottom_up_features[i + 1].size()[-1]))(x)
                except IndexError:
                    return pyramid_feature_maps


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_features: int):
        super().__init__()

        self.in_features = in_features

        self.bottom_up_path = BottomUpPathway(self.in_features)
        self.top_down_path = TopDownPathway(top_features=512, in_features=256)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.bottom_up_path(x)
        x = self.top_down_path(x)

        return x


class FPNClassifier(FeaturePyramidNetwork):

    def __init__(self, in_features: int):
        super().__init__(in_features=in_features)
        self.fpn = nn.Sequential(
            self.bottom_up_path,
            self.top_down_path
        )

        self.fc = nn.Sequential(
            StackPyramidPlanes(),
            Flatten(),

            nn.Linear(1024, 1024),
            nn.SELU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fpn(x)
        x = self.fc(x)

        return x


def accuracy(model: nn.Module, test_loader: DataLoader):
    with torch.no_grad():
        model.eval()

        correct = 0
        for inputs, target in test_loader:
            if round(model(inputs).item()) == round(target.item()):
                correct += 1
        return correct/len(test_loader.dataset)

