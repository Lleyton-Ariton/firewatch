import os

import numpy as np
import firewatch.src.utils.multicore as multi

from PIL import Image
from typing import *


def load_images(data_path: str=False) -> Tuple[List[Image.Image], List[Image.Image]]:

    if not data_path:
        data_path = '/'.join(os.getcwd().split('/')[:-1] + ['data/classification'])

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
    with multi.RayManager():
        for category in image_loader():
            resized.append(multi.multicore(resize, category))
    return resized


def images_to_array(images: List[Image.Image]) -> np.ndarray:
    return [np.array(img) for img in images]
