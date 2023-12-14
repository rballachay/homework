import os
from functools import wraps
from pathlib import Path
import pickle
import json


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
