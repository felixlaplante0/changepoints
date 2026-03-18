from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

# TypeVars
T = TypeVar("T")


def np_cache(func: Callable[..., T]) -> Callable[..., T]:
    cache: dict[tuple[int, ...], T] = {}

    def wrapper(*args: Any) -> T:
        key = tuple(hash(arr.tobytes()) for arr in args if isinstance(arr, np.ndarray))
        if key not in cache:
            cache[key] = func(*args)

        return cache[key]

    return wrapper
