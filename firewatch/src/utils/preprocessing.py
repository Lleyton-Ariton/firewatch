import os
import abc
import ray

import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas as pd
import pandas_read_xml as pdx

import random
import pathlib
import multiprocessing as mp

from typing import *


class DataProcessor(metaclass=abc.ABCMeta):

    __IGNORES = ['.DS_Store']

    @classmethod
    def get_ignores(cls):
        return cls.__IGNORES

    @classmethod
    def set_ignores(cls, ignores: List[str]) -> None:
        if isinstance(ignores, list):
            cls.__IGNORES = ignores
            return None
        raise TypeError()

    @classmethod
    def add_ignore(cls, ignore: [str, List[str]]) -> None:
        if isinstance(ignore, list):
            cls.__IGNORES.extend(ignore)
            return None

        elif isinstance(ignore, str):
            cls.__IGNORES.append(ignore)
            return None
        raise TypeError()

    @classmethod
    def remove_ignore(cls, ignore: str) -> None:
        if isinstance(ignore, str):
            cls.__IGNORES.remove(ignore)
            return None
        raise TypeError()

    def __init__(self, data_root_path: str=False, image_size: Tuple[int, int]=(500, 500)):
        self.__data_root_path = data_root_path

        self.smoke = []
        self.no_smoke = []

        self._image_size = image_size

    @property
    def root_path(self) -> str:
        return self.__data_root_path

    def get_root_path(self) -> str:
        return self.__data_root_path

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def get_image_size(self) -> Tuple[int, int]:
        return self._image_size

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        if isinstance(image_size, tuple) and len(image_size) < 2:
            self._image_size = image_size
            return None
        raise TypeError()

    @abc.abstractmethod
    def preprocess(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class SmokeClassificationDataProcessor(DataProcessor):

    __IGNORES = ['.DS_Store']

    @classmethod
    def get_ignores(cls):
        return cls.__IGNORES

    @classmethod
    def set_ignores(cls, ignores: List[str]) -> None:
        if isinstance(ignores, list):
            cls.__IGNORES = ignores
            return None
        raise TypeError()

    @classmethod
    def add_ignore(cls, ignore: [str, List[str]]) -> None:
        if isinstance(ignore, list):
            cls.__IGNORES.extend(ignore)
            return None

        elif isinstance(ignore, str):
            cls.__IGNORES.append(ignore)
            return None
        raise TypeError()

    @classmethod
    def remove_ignore(cls, ignore: str) -> None:
        if isinstance(ignore, str):
            cls.__IGNORES.remove(ignore)
            return None
        raise TypeError()

    def __init__(self, data_root_path: str=False, image_size: Tuple[int, int]=(500, 500)):
        self.__data_root_path = data_root_path
        if not self.__data_root_path:
            self.__data_root_path = str(pathlib.Path(__file__).parent.absolute())
            self.__data_root_path = self.__data_root_path.split('/')[:-3] + ['data/classification']
            self.__data_root_path = '/'.join(self.__data_root_path)

            super().__init__(data_root_path=self.__data_root_path, image_size=image_size)
        super().__init__(data_root_path=data_root_path, image_size=image_size)

    @property
    def root_path(self) -> str:
        return self.__data_root_path

    def get_root_path(self) -> str:
        return self.__data_root_path

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def get_image_size(self) -> Tuple[int, int]:
        return self._image_size

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        if isinstance(image_size, tuple) and len(image_size) < 3:
            self._image_size = image_size
            return None
        raise TypeError()

    def preprocess(self, image_size: Tuple[int, int]=False, automatic_initialization: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        if image_size:
            self.set_image_size(image_size)

        categories = filter(lambda x: x not in self.get_ignores(), sorted(os.listdir(self.__data_root_path)))
        categories = filter(lambda x: not os.path.isdir(x), categories)

        images = []

        if automatic_initialization:
            ray.init()

        for category in categories:
            category_path = os.listdir(self.get_root_path() + f'/{category}')

            it = ray.util.iter.from_items(category_path, num_shards=mp.cpu_count())
            it = it.for_each(
                lambda x: cv2.imread(f'{self.get_root_path()}/{category}/{x}')
            ).for_each(
                lambda x: cv2.resize(x, self.get_image_size())
            )

            images.append([image for image in it.gather_sync()])

        if ray.is_initialized() and automatic_initialization:
            ray.shutdown()

        no_smoke, smoke = map(np.array, images)
        self.smoke, self.no_smoke = smoke, no_smoke

        return self.smoke, self.no_smoke


def grid_data_processor(data_root_path: str, image_size: Tuple[int, int]=(227, 170)):
    return SmokeClassificationDataProcessor(data_root_path=data_root_path).preprocess(image_size=image_size)


class SmokeClassificationDataset(Dataset):

    @staticmethod
    def shuffle_xy(x: [List, torch.Tensor, np.ndarray],
                   y: [List, torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:

        merged = list(zip(x.tolist(), y.tolist()))
        random.shuffle(merged)

        x, y = zip(*merged)
        return torch.tensor(x).float(), torch.tensor(y).float()

    def __init__(self, data_root: str=False, image_size: Tuple[int, int]=(500, 500),
                 shuffle: bool=True, automatic_initialization: bool=False):
        super().__init__()

        self.__data_root = data_root
        self._image_size = image_size
        self.shuffle = shuffle
        self.automatic_initialization = automatic_initialization

        processor = SmokeClassificationDataProcessor(data_root_path=self.__data_root, image_size=self._image_size)

        smoke, no_smoke = processor.preprocess(automatic_initialization=automatic_initialization)
        smoke, no_smoke = smoke.reshape((len(smoke), 3, *self._image_size)), \
                          no_smoke.reshape((len(no_smoke), 3, *self._image_size))

        x_data = np.concatenate((smoke, no_smoke))
        y_data = np.concatenate((np.ones(shape=(smoke.shape[0], 1)),
                                 np.zeros(shape=(no_smoke.shape[0], 1))))

        self.x_data, self.y_data = torch.from_numpy(x_data).float(), torch.from_numpy(y_data).float()

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self) -> int:
        return len(self.x_data)
