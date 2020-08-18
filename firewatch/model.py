import ray
from ray.util.sgd import TorchTrainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import *
from firewatch.src.utils.multicore import RayManager
from firewatch.src.utils.experimental import SmokeClassificationDataset
from firewatch.src.utils import multicore as multi

import time


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x)


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
            nn.BatchNorm2d(512)  # New
        )

        self.dense_block = nn.Sequential(  # New
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


def accuracy(model: nn.Module, test_loader: DataLoader):
    with torch.no_grad():
        model.eval()

        correct = 0
        for inputs, target in test_loader:
            if round(model(inputs).item()) == round(target.item()):
                correct += 1
        return correct/len(test_loader.dataset)


if __name__ == '__main__':

    smoke_net = SmokeClassifier()

    smoke_net.load_state_dict(torch.load('/Users/andreeaariton/PycharmProjects/firewatch/data/SmokeNet-15[v1]',
                                         map_location=torch.device('cpu')))

    smoke_test_loader = DataLoader(
        SmokeClassificationDataset(data_root='/Users/andreeaariton/PycharmProjects/firewatch/data/grid_version/validate',
                                   automatic_initialization=True, image_size=(227, 170)),
        shuffle=True)

    print(accuracy(smoke_net, smoke_test_loader))
