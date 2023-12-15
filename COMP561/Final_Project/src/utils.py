import os
from functools import wraps
from pathlib import Path
import numpy as np
import json


def scoring_function(prob: float):
    if prob == 0:
        prob = 1e-10
    score = np.emath.logn(4, prob) + 1
    return score


def cachewrapper(path: Path):
    """caching decorator to save intermediate dfs,
    makes repeated testing much faster
    """
    if isinstance(path, Path):
        path = str(path)

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            use_cache = True
            if os.path.exists(path) and use_cache:
                with open(path, "rb") as obj:
                    data = json.load(obj)
            else:
                data = function(*args, **kwargs)
                with open(path, "w") as obj:
                    json.dump(data, obj)
            return data

        return wrapper

    return decorator
