import os

import torch
from torch.utils.data import Dataset

import numpy as np
import firewatch.src.utils.multicore as multi

import random
from PIL import Image
from typing import *


IMAGE_SIZE = (500, 500)


def load_images(data_path: str=False) -> Tuple[List[Image.Image], List[Image.Image]]:

    if not data_path:
        data_path = '/'.join(os.getcwd().split('/')[:-2] + ['data/classification'])

    smoke, no_smoke = [], []
    for name in ['smoke', 'no_smoke']:
        dir_path = f'{data_path}{"/"}{name}'

        for image_path in os.listdir(dir_path):
            img = Image.open(dir_path + '/' + image_path)
            if name == 'smoke':
                smoke.append(img)
            else:
                no_smoke.append(img)
    return smoke, no_smoke


def resize_images(size: Tuple[int, int], image_loader: Callable=load_images):

    def resize(image):
        return image.resize(size, Image.ANTIALIAS)

    resized = []
    for category in image_loader():
        resized.append(multi.multicore(resize, category))
    return resized


def images_to_array(images: List[Image.Image]) -> List[np.ndarray]:
    return [np.array(img) for img in images]


def preprocess(size: Tuple[int, int]=IMAGE_SIZE,
               image_loader: Callable=load_images) -> Tuple[torch.Tensor, torch.Tensor]:
    smoke, no_smoke = map(np.array, map(images_to_array, resize_images(size=size, image_loader=image_loader)))
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


class SmokeDataset(Dataset):

    def __init__(self, size: Tuple[int, int]=IMAGE_SIZE, image_loader: Callable=load_images):
        self.x_data, self.y_data = shuffle_xy(*preprocess(size=size, image_loader=image_loader))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[index], self.y_data[index]

    def __len__(self) -> int:
        return len(self.x_data)
