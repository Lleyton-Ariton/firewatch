import ray
import multiprocessing as mp

from typing import *


class RayManager:

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def __enter__(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(*self.__args, **self.__kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        ray.shutdown()


def partition(parting_size: int, list1: List) -> List[List]:
    for i in range(0, len(list1), parting_size):
        yield list1[i:i + parting_size]


def multicore(func: Callable, iter1: List) -> Any:
    results = []

    func = ray.remote(func)
    for part in partition(mp.cpu_count(), iter1):
        futures = [func.remote(element) for element in part]
        results.extend(ray.get(futures))
    return results
