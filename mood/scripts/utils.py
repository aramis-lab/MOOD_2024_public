from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import nibabel as nib
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from clinicadl.utils.caps_dataset.data import return_dataset
from clinicadl.utils.maps_manager import MapsManager
from clinicadl.utils.maps_manager.maps_manager_utils import read_json
from mood.data.concat_dataset import CapsPairedDataset


def get_dataloader(
    maps_manager: MapsManager,
    output_name: str,
    preprocessing: Path,
    data_group: str,
    split: int,
    n_proc: int,
    batch_size: int,
    selection_metrics: str,
) -> CapsPairedDataset:
    group_df, group_parameters = maps_manager.get_group_info(data_group, split)

    preprocessing_in_dict = read_json(preprocessing)
    preprocessing_out_dict = deepcopy(preprocessing_in_dict)
    preprocessing_gt_dict = deepcopy(preprocessing_in_dict)

    preprocessing_in_dict["file_type"]["pattern"] = preprocessing_out_dict["file_type"][
        "pattern"
    ].replace("input", "mood")
    preprocessing_out_dict["file_type"]["pattern"] = preprocessing_out_dict[
        "file_type"
    ]["pattern"].replace("input", output_name)
    preprocessing_gt_dict["file_type"]["pattern"] = preprocessing_gt_dict["file_type"][
        "pattern"
    ].replace("input", "mask")

    preprocessing_in_dict["preprocessing"] = "custom"
    preprocessing_out_dict["preprocessing"] = "custom"
    preprocessing_gt_dict["preprocessing"] = "custom"

    caps_pred_dir = Path(
        maps_manager.maps_path,
        f"split-{split}",
        f"best-{selection_metrics}",
        "CapsOutput",
    )
    caps_gt_dir = group_parameters["caps_directory"]

    input_dataset = return_dataset(
        caps_gt_dir,
        group_df,
        preprocessing_in_dict,
        all_transformations=None,
        multi_cohort=group_parameters["multi_cohort"],
    )
    output_dataset = return_dataset(
        caps_pred_dir,
        group_df,
        preprocessing_out_dict,
        all_transformations=None,
        multi_cohort=group_parameters["multi_cohort"],
    )
    gt_dataset = return_dataset(
        caps_gt_dir,
        group_df,
        preprocessing_gt_dict,
        all_transformations=None,
        multi_cohort=group_parameters["multi_cohort"],
    )
    stacked_dataset = CapsPairedDataset(
        gt=gt_dataset, input=input_dataset, output=output_dataset
    )
    dataloader = DataLoader(
        stacked_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_proc,
    )

    return dataloader


class ImageSaver:
    """
    To parallelize image-saving.

    Parameters
    ----------
    prefix : Path
        The folder where to save images.
    suffix : str
        The suffix at the end of the images' names.
    n_proc : int (optional, default=2)
        To parallelize.
    """

    def __init__(self, prefix: Path, suffix: str, n_proc: int = 2):
        self.prefix = prefix
        self.suffix = suffix
        self.n_proc = n_proc

    def save_batch(self, img_batch: Dict[str, Iterable]) -> None:
        """
        Saves a batch of image.

        Parameters
        ----------
        img_batch : Dict[str, Iterable]
            The batch with the data and other information
            (e.g. participant_id, session_id).
        """
        img_batch = [dict(zip(img_batch, t)) for t in zip(*img_batch.values())]
        multi_pool = mp.Pool(processes=self.n_proc)
        outputs = multi_pool.map(
            partial(save_img, prefix=self.prefix, suffix=self.suffix), img_batch
        )
        multi_pool.close()
        multi_pool.join()

        return np.array(outputs)


def save_img(img: Dict[str, Any], prefix: Path, suffix: str) -> None:
    """
    Saves an image, given with information (e.g. session_id, participant_id).

    Parameters
    ----------
    img : np.ndarray
        The image and the information.
    prefix : Path
        The folder where to save images (e.g. a caps dataset).
    suffix : str
        The suffix at the end of the images' names.
    """
    participant_id = img["participant_id"]
    session_id = img["session_id"]
    save_in = (
        prefix
        / "subjects"
        / participant_id
        / session_id
        / "custom"
        / f"{participant_id}_{session_id}_{suffix}.nii.gz"
    )
    nib.save(nib.Nifti1Image(img["data"], np.eye(4)), save_in)
    print(f"Saved in {save_in}")


def mean_and_confidence(series: list) -> Tuple[float, float]:
    """
    Computes 99% confidence interval.

    Parameters
    ----------
    series : list
        The samples.

    Returns
    -------
    Tuple[float, float]
        The mean and the confidence interval.
    """
    mean = np.mean(series)
    std = np.std(series)
    conf = 2.576 * std / np.sqrt(len(series))

    return mean, conf
