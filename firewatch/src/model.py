import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from typing import *

from firewatch.src.utils import preprocessing


IMAGE_SIZE = (500, 500)


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


def preprocess(size: Tuple[int, int]=IMAGE_SIZE,
               image_loader: Callable=preprocessing.load_images) -> Tuple[torch.Tensor, torch.Tensor]:
    smoke, no_smoke = map(np.array, map(preprocessing.images_to_array,
                                        preprocessing.resize_images(size=size,
                                                                    image_loader=image_loader)))
    smoke = smoke.reshape((len(smoke), 3, *size))
    no_smoke = no_smoke.reshape((len(no_smoke), 3, *size))

    x_data = np.concatenate((smoke, no_smoke))
    y_data = np.concatenate((np.ones(shape=(smoke.shape[0], 1)),
                             np.zeros(shape=(no_smoke.shape[0], 1))))
    return torch.from_numpy(x_data), torch.from_numpy(y_data)


def shuffle_xy(x: Union[List, torch.Tensor, np.ndarray],
               y: Union[List, torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:

    merged = list(zip(x.tolist(), y.tolist()))
    random.shuffle(merged)

    x, y = zip(*merged)
    return torch.tensor(x).float(), torch.tensor(y).float()


if __name__ == '__main__':

    # TESTING PURPOSES

    smoke_net = SmokeClassifier()
    x_train, y_train = shuffle_xy(*preprocess())

    if torch.cuda.is_available():
        smoke_net.cuda()

        x_train.cuda(), y_train.cuda()

    epochs = 2
    optimizer = optim.SGD(smoke_net.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for xbatch, ybatch, in zip(x_train, y_train):
            xbatch = xbatch.view(1, 3, *IMAGE_SIZE)

            out = smoke_net(xbatch)
            loss = criterion(out, ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())
