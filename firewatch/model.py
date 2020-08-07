import ray
from ray.util.sgd import TorchTrainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import *
from firewatch.src.utils.multicore import RayManager
from firewatch.src.utils.preprocessing import IMAGE_SIZE, SmokeDataset
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
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=3),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        )

        self.pool1 = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv_block128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1)
        )

        self.pool2 = nn.MaxPool2d(2, 2)
        self.norm2 = nn.BatchNorm2d(128)

        self.conv_block256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(2, 2)
        self.norm3 = nn.BatchNorm2d(256)

        self.conv_block512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.ReLU()
        )

        self.pool4 = nn.MaxPool2d(2, 2)
        self.norm4 = nn.BatchNorm2d(512)

        self.flatten = Flatten()

        self.dense1 = nn.Linear(512 * 3 * 3, 500)
        self.dense2 = nn.Linear(500, 100)
        self.dense3 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block64(x)
        x = self.pool1(x)
        x = self.norm1(x)

        x = self.conv_block128(x)
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.conv_block256(x)
        x = self.pool3(x)
        x = self.norm3(x)

        x = self.conv_block512(x)
        x = self.pool4(x)
        x = self.norm4(x)

        x = self.flatten(x)
        for dense_layer in [self.dense1, self.dense2]:
            x = torch.relu(dense_layer(x))
        x = torch.sigmoid(self.dense3(x))

        return x


def model_creator(config: Dict) -> nn.Module:
    return SmokeClassifier()


def optimizer_creator(model: nn.Module, config: Dict) -> Any:
    return torch.optim.SGD(model.parameters(), lr=config.get('lr', 0.0001))


def loss_creator(config: Dict) -> Any:
    return nn.BCELoss()


if __name__ == '__main__':

    # TESTING PURPOSES

    start = time.time()

    with RayManager():
        TRAIN_DATASET = SmokeDataset(size=IMAGE_SIZE)
    end = time.time()

    print('Loaded Dataset [time: {}]'.format(end-start))

    def data_creator(config: Dict) -> DataLoader:
        return DataLoader(TRAIN_DATASET, batch_size=config.get('batch_size', 1))


    with RayManager():
        print('Training Start')

        trainer = TorchTrainer(
            model_creator=model_creator,
            data_creator=data_creator,
            optimizer_creator=optimizer_creator,
            loss_creator=loss_creator,
            config={'lr': 0.001},
            num_workers=multi.mp.cpu_count(),
            use_tqdm=True
        )

        for _ in range(1):
            metrics = trainer.train()
            print(metrics)

        trainer.shutdown()
