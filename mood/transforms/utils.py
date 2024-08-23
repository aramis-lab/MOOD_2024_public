import random
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from typing import ContextManager, Generator, Optional, Tuple, Union

#from mood.utils.exceptions import InvalidPathException
import nibabel as nib
import numpy as np


class TransformsSampleLevel(str, Enum):
    GLOBAL_BLURRING = "GlobalBlurring"
    GLOBAL_ELASTIC_DEFORMATION = "GlobalElasticDeformation"
    GHOSTING = "Ghosting"
    SPIKE = "Spike"
    BIAS_FIELD = "BiasField"
    NOISE = "Noise"
    IMAGE_COMPOSE = "ImageCompose"


class TransformsPixelLevel(str, Enum):
    CORRUPT_SLICE = "CorruptSlice"
    LOCAL_PIXEL_SHUFFLIG = "LocalPixelShuffling"
    DARK_OR_BRIGHT = "DarkOrBright"
    LOCAL_BLURRING = "LocalBlurring"
    LOCAL_ELASTIC_DEFORMATION = "LocalElasticDeformation"
    LOCAL_COMPOSE = "LocalCompose"


@contextmanager
def set_seed(seed: int) -> Generator[None, None, None]:
    """
    Sets Python's and Numpy's seed.

    Parameters
    ----------
    seed : int

    Yields
    ------
    Generator[None, None, None]
    """
    import torch.random as torch_random

    torch_state = torch_random.get_rng_state()
    np_state = np.random.get_state()
    state = random.getstate()
    try:
        torch_random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        torch_random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(state)


def set_randomness(seed: Optional[int]) -> ContextManager:
    """
    Creates a context manager to set the seeds locally.

    Parameters
    ----------
    seed : Optional[int]

    Returns
    -------
    ContextManager
    """
    if seed is not None:
        return set_seed(seed)
    else:
        return nullcontext()


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Saves a 3D image to NIfTI format.

    Parameters
    ----------
    tensor : np.ndarray
    path : Union[str, Path]
    """
    assert ".nii.gz" in str(path), "Must be a .nii.gz file."
    nifti = nib.Nifti1Image(image.astype(np.float32), np.eye(4))
    nib.save(nifti, path)


def extract_image(path: Union[str, Path]) -> np.ndarray:
    """
    Extracts a 3D image from a NIfTI file.

    Parameters
    ----------
    path : Union[str, Path]

    Returns
    -------
    np.ndarray
    """
    if not Path(path).is_file():
        raise ValueError(f"the path {path}is not a file")
    image_array = nib.load(path).get_fdata(dtype="float32")

    return image_array


def resize(img: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
    """
    Resizes a 3D image with trilinear interpolation.

    Parameters
    ----------
    img : np.ndarray
    size : Tuple[int, int, int]

    Returns
    -------
    np.ndarray
    """
    from torch import from_numpy
    from torch.nn.functional import interpolate

    output = interpolate(
        from_numpy(img)[None, None, :, :, :], size=size, mode="trilinear"
    )
    return output.squeeze((0, 1)).numpy().astype(img.dtype)
