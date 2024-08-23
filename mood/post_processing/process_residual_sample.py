
from typing import Callable, Optional
from pathlib import Path
import os
import nibabel as nib
import numpy as np
import torch.multiprocessing as mp
from skimage.filters import median
from enum import Enum
from mood.transforms.utils import save_image, extract_image
from mood.utils.exceptions import InvalidArgumentException, InvalidPathException
from clinicadl.utils.clinica_utils import clinicadl_file_reader, find_sub_ses_pattern_path, get_subject_session_list


class ResidualType(str, Enum):

    VAL = "validation"
    INPUT = "input"
    CLASSIC = "classic"
    SSIM = "ssim"


def compute_residual(residual_np: np.ndarray, residual_type: ResidualType, residual_std_np: np.ndarray, residual_mean_np: Optional[np.ndarray] = None) -> np.ndarray:
    if residual_type == ResidualType.VAL:
        tmp = (residual_np - residual_mean_np) / residual_std_np
        tmp[tmp == np.inf] = 0
        return tmp
    elif residual_type == ResidualType.INPUT:
        tmp = residual_np / residual_std_np
        tmp[tmp == np.inf] = 0
        return tmp
    else:
        raise InvalidArgumentException()


def create_residual(caps_input_path : Path, residual_type: ResidualType, maps_path: Path, split: int, shapes = (256, 256, 256)):
    caps_output_path = maps_path / f"split-{split}" / "best-loss" / "CapsOutput"
    residual_path = caps_output_path.parent / "residual"

    if not residual_path.is_dir():
        residual_path.mkdir(parents = True, exist_ok = True)

    if residual_type == ResidualType.VAL:
        sub_ses_tsv = maps_path / "groups" / "validation" / f"split-{split}" / "data.tsv"

    elif residual_type == ResidualType.INPUT:
        sub_ses_tsv = maps_path / "groups" / "train+validation.tsv"
    
    elif residual_type == ResidualType.CLASSIC:
        sub_ses_tsv = None

    subjects, sessions = get_subject_session_list(input_dir = caps_input_path, subject_session_file = sub_ses_tsv, is_bids_dir = False)
    path_list, _ = clinicadl_file_reader(subjects, sessions, caps_output_path, information= {"pattern": "*custom/*_residual.nii.gz", "description": "pattern for mood 2024 challenge"})

    residual_out = np.zeros(shapes) 

    for path in path_list:
        residual_array = extract_image(path)
        residual_out +=residual_array

    mean_array =  residual_out/len(path_list) #sum_ #np.mean(residual_list,  axis=0)
    save_image(mean_array, residual_path / f"{residual_type.value}_mean.nii.gz" )

    print(f"residual mean for {residual_type.value} is saved at: ", residual_path / f"{residual_type.value}_mean.nii.gz")

    residual_out_std = np.zeros(shapes) 
    for i, path in enumerate(path_list):
        residual_array = extract_image(path)
        residual_out_std += (residual_array - mean_array)**2

    std_array = np.sqrt(residual_out_std /len(path_list))  #np.std(residual_list, axis=0)
    save_image(std_array, residual_path / f"{residual_type.value}_std.nii.gz" )

    print(f"residual std for {residual_type.value} is saved at: ", residual_path / f"{residual_type.value}_std.nii.gz")



def get_residual(residual_np: np.ndarray, residual_type: ResidualType, caps_output_path: Path):

    """
    Get the residual computed with input or validation mean and/or std.  

    Parameters
    ----------
    residual_np : np.ndarray
        The batch of residuals (4D array).
    residual_type : 

    maps_path :  Path to the 
    Returns
    -------
    np.ndarray
        The postprocessed batch.
    """
    try: 
        residual_type = ResidualType(residual_type)
    except:
        raise InvalidArgumentException()

    if residual_type == ResidualType.VAL or residual_type == ResidualType.INPUT:
        residual_path = caps_output_path.parent / "residual"

        if not residual_path.is_dir() or not any(residual_path.iterdir()):
            raise InvalidPathException()

        if residual_type == ResidualType.VAL:
            residual_mean_np = extract_image(residual_path / "validation_mean.nii.gz")
            residual_mean_out = (residual_mean_np - residual_mean_np.min()) / (residual_mean_np.max() - residual_mean_np.min())

            residual_std_np = extract_image(residual_path / "validation_std.nii.gz")
            residual_std_out = (residual_std_np - residual_std_np.min()) / (residual_std_np.max() - residual_std_np.min())


        elif residual_type == ResidualType.INPUT:
            residual_mean_np = None
            residual_std_np = extract_image(residual_path / "input_std.nii.gz")

        residual_out = compute_residual(residual_np = residual_np, residual_mean_np = residual_mean_out, residual_std_np = residual_std_out, residual_type = residual_type)
    
    else:
        residual_out = residual_np

    return residual_out


def post_processing(input_np: np.ndarray, output_np: np.ndarray, mean_np: np.ndarray, std_np: np.ndarray):
    tmp_residual_np = output_np - input_np
    
    with np.errstate(divide='ignore', invalid='ignore'):
        residual_tmp = np.true_divide((np.abs(tmp_residual_np) - np.abs(mean_np)),np.abs(std_np))
        residual_tmp[residual_tmp == np.inf] = 0
        residual_tmp = np.nan_to_num(residual_tmp, nan=0.0)

    # tmp_residual_np = (tmp_residual_np - tmp_residual_np.min()) / (tmp_residual_np.max() - tmp_residual_np.min()) #??

    # residual_np = (tmp_residual_np - mean_np) / std_np
    # residual_np[residual_np == np.inf] = 0
    
    # residual_np = (residual_np - residual_np.min()) / (residual_np.max() - residual_np.min())
    
    return residual_tmp