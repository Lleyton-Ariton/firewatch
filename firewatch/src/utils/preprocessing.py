import os
import abc
import ray

import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas_read_xml as pdx

from sklearn.model_selection import train_test_split

import image_bbox_slicer as ibs

import shutil
import random
import pathlib
import itertools
import multiprocessing as mp

from typing import *


DEFAULT_SOURCES = []


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

    def __call__(self, **kwargs):
        return self.preprocess(**kwargs)


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

    def preprocess(self, image_size: Tuple[int, int]=False, automatic_initialization: bool=True,
                   augment_data: bool=False) -> Tuple[np.ndarray, np.ndarray]:
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

            if augment_data:
                it = it.for_each(lambda x: (x, np.flip(x, axis=1)))
                images.append([image for image in itertools.chain(*it.gather_sync())])
            else:
                images.append([image for image in it.gather_sync()])

        if ray.is_initialized() and automatic_initialization:
            ray.shutdown()

        no_smoke, smoke = map(np.array, images)
        self.smoke, self.no_smoke = smoke, no_smoke

        return self.smoke, self.no_smoke


def grid_data_processor(data_root_path: str, image_size: Tuple[int, int]=(227, 170)):
    return SmokeClassificationDataProcessor(data_root_path=data_root_path).preprocess(image_size=image_size)


class SmokeLocalizationBBoxProcessor(DataProcessor):

    def __init__(self, data_root_path: str=None, image_size: Tuple[int, int]=(227, 170)):
        self.__data_root_path = data_root_path
        if not self.__data_root_path:
            self.__data_root_path = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3]) + '/data/sliced'

            super().__init__(data_root_path=self.__data_root_path, image_size=image_size)
        super().__init__(data_root_path=data_root_path, image_size=image_size)

    def preprocess(self, **kwargs):
        pass  # Not Finished


class SmokeGridBBoxProcessor(DataProcessor):

    def __init__(self, data_root_path: str=None, image_size: Tuple[int, int]=(227, 170)):
        self.__data_root_path = data_root_path
        if not self.__data_root_path:
            self.__data_root_path = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3]) + '/data/sliced'

            super().__init__(data_root_path=self.__data_root_path, image_size=image_size)
        super().__init__(data_root_path=data_root_path, image_size=image_size)

    def preprocess(self, **kwargs):
        pass  # Not Finished


class SmokeClassificationDataset(Dataset):

    @staticmethod
    def shuffle_xy(x: [List, torch.Tensor, np.ndarray],
                   y: [List, torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:

        merged = list(zip(x.tolist(), y.tolist()))
        random.shuffle(merged)

        x, y = zip(*merged)
        return torch.tensor(x).float(), torch.tensor(y).float()

    def __init__(self, data_root: str=False, image_size: Tuple[int, int]=(227, 170),
                 shuffle: bool=True, automatic_initialization: bool=False, augment_data: bool=False):
        super().__init__()

        self.__data_root = data_root
        self._image_size = image_size
        self.shuffle = shuffle
        self.automatic_initialization = automatic_initialization

        processor = SmokeClassificationDataProcessor(data_root_path=self.__data_root, image_size=self._image_size)

        smoke, no_smoke = processor.preprocess(automatic_initialization=automatic_initialization,
                                               augment_data=augment_data)
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


def slice_bbox_images(image_source: str=None, annotation_source: str=None,
                      sliced_image_dir: str=None, sliced_annotation_dir: str=None,
                      keep_partial_labels: bool=True, ignore_empty_files: bool=False,
                      slice_by_number=9, slice_by_size: Tuple=False) -> None:

    if image_source is None:
        image_source = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3])
        image_source += '/data/localization/images'

    if annotation_source is None:
        annotation_source = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3])
        annotation_source += '/data/localization/annotations'

    if sliced_image_dir is None:
        sliced_image_dir = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3])
        sliced_image_dir += '/data/sliced/sliced_images'

    if sliced_annotation_dir is None:
        sliced_annotation_dir = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3])
        sliced_annotation_dir += '/data/sliced/sliced_annotations'

    slicer = ibs.Slicer()
    slicer.config_dirs(img_src=image_source, ann_src=annotation_source,
                       img_dst=sliced_image_dir, ann_dst=sliced_annotation_dir)

    slicer.keep_partial_labels = keep_partial_labels
    slicer.ignore_empty_tiles = ignore_empty_files

    if slice_by_number:
        slicer.slice_by_number(number_tiles=slice_by_number)
        return

    slicer.slice_by_size(tile_size=slice_by_size)
    return


class _ExpandGridWithSlicerData(DataProcessor):

    __WAS_CALLED = 0

    @classmethod
    def is_called(cls) -> None:
        cls.__WAS_CALLED += 1
        return None

    @classmethod
    def get_calls(cls) -> int:
        return cls.__WAS_CALLED

    @classmethod
    def was_called(cls) -> bool:
        return True if cls.was_called() > 0 else False

    def __init__(self, data_root_path: str=False, target_dir: str=False,
                 slicer_annotations_dir: str='sliced_annotations', slicer_images_dir: str='sliced_images',
                 validation_split: float=0.3, max_emptytile_inbalance_factor: float=1.25,
                 image_size: Tuple[int, int]=(227, 170)):

        self.__data_root_path, self._target_dir = data_root_path, target_dir

        if not self.__data_root_path:
            self.__data_root_path = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3]) + '/data/sliced'
        if not self._target_dir:
            self._target_dir = '/'.join(str(pathlib.Path.cwd()).split('/')[:-3]) + '/data/grid_version'

            super().__init__(data_root_path=self.__data_root_path, image_size=image_size)
        super().__init__(data_root_path=self.__data_root_path, image_size=image_size)

        self.slicer_annotations_dir = slicer_annotations_dir
        self.slicer_images_dir = slicer_images_dir

        self.validation_split = validation_split
        if self.validation_split <= 0.0:
            self.validation_split = None

        self.max_emptytile_inbalance_factor = max_emptytile_inbalance_factor

    def preprocess(self, image_size: Tuple[int, int]=False,
                   automatic_initialization: bool=True, **kwargs) -> None:

        if not os.path.exists(self.get_root_path()):
            _slice_bbox_images(image_source=kwargs.get('image_source', None),
                               annotation_source=kwargs.get('annotations_source', None),
                               sliced_image_dir=kwargs.get('sliced_image_dir', None),
                               sliced_annotation_dir=kwargs.get('sliced_image_dir', None),
                               keep_partial_labels=kwargs.get('keep_partial_labels', True),
                               ignore_empty_files=kwargs.get('ignore_empty_files', False))

        if image_size:
            self.set_image_size(image_size)

        xml_root = f'{self.get_root_path()}/{self.slicer_annotations_dir}'
        xml_files = os.listdir(xml_root)

        images_with_smoke = []
        images_without_smoke = []

        with_counter = 0
        without_counter = 0
        for xml_file in xml_files:
            xml = pdx.read_xml(f'{xml_root}/{xml_file}', transpose=True)
            try:
                xml['annotation']['object']['bndbox']

                path = xml['annotation']['path'] + '.jpeg'
                path = f'{self.get_root_path()}/{self.slicer_images_dir}/{path.split("/")[-1:][0]}'
                images_with_smoke.append(path)

                with_counter += 1
            except KeyError:
                if without_counter < with_counter * self.max_emptytile_inbalance_factor:
                    path = xml['annotation']['path'] + '.jpeg'
                    path = f'{self.get_root_path()}/{self.slicer_images_dir}/{path.split("/")[-1:][0]}'
                    images_without_smoke.append(path)

                    without_counter += 1

        if self.validation_split is not None:
            for images, category in zip([images_with_smoke, images_without_smoke], ['grid_smoke', 'grid_no_smoke']):
                train_images, test_images = train_test_split(images, test_size=self.validation_split,
                                                             shuffle=kwargs.get('shuffle', True))

                for image_set, section in zip([train_images, test_images], ['train', 'validate']):
                    for image in image_set:
                        shutil.move(image, f'{self._target_dir}/{section}/{category}/{image.split("/")[-1:][0]}')

        return None
