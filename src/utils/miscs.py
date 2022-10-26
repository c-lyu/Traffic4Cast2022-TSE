import os, sys
import torch
import numpy as np
from pathlib import Path
from functools import partial


def config_sys_path(root_dir="."):
    root_dir = Path(root_dir)
    sys.path.insert(0, os.path.abspath(root_dir / "NeurIPS2022-traffic4cast"))
    sys.path.insert(1, os.path.abspath("."))


def fill_zero(*args, **kwargs):
    """Filling invalid values with 0."""
    f = partial(np.nan_to_num, nan=0.0, posinf=0.0, neginf=0.0)
    return f(*args, **kwargs)


def check_torch_tensor(array):
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise TypeError(f"Array must be torch.Tensor or np.ndarray, got {type(array)}.")


def check_test_size(s):
    """Util to transform test_size.

    If s is a float smaller than 1, the test data will be sampled by a porportion of s.
    If s is a integer larger than 1, the test data will be sampled by a number of s.
    """
    if s < 1:
        return s
    else:
        return int(s)
