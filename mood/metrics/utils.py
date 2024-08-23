from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as find_label


def get_image_confusion_matrix(
    anomaly_map: np.ndarray,
    seg_objects: np.ndarray,
    # bin_thres: float,
    size_thres: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Find the number of TPs, FPs and FNs for an image.

    Parameters
    ----------
    anomaly_map : np.ndarray
        The binarized prediction.
    seg_objects : np.ndarray
        The ground truth mask.
    size_thres : Optional[int] (optional, default=None)
        The minimum anomaly volume.

    Returns
    -------
    Tuple[int, int, int]
        Number of TPs, FPs and FNs in the image.
    """
    # pred_thres = anomaly_map > bin_thres
    pred_thres = anomaly_map

    pred_labeled, n_labels = find_label(pred_thres, np.ones((3, 3, 3)))
    seg_labeled, _ = find_label(seg_objects, np.ones((3, 3, 3)))

    label_counts = np.bincount(pred_labeled.flatten())

    matched_dict = defaultdict(bool)
    fp = 0
    fn = 0
    tp = 0

    for lbl_idx in range(1, n_labels + 1):
        if (size_thres is not None) and (label_counts[lbl_idx] < size_thres):
            continue
        pred_thres_copy = pred_thres.copy()
        pred_thres_copy[pred_labeled != lbl_idx] = 0
        x, y, z = center_of_mass(pred_thres_copy)

        x, y, z = int(x), int(y), int(z)

        if seg_objects[x, y, z] != 0:
            gt_sum = np.sum(seg_labeled == seg_labeled[x, y, z])
            pred_sum = np.sum(pred_thres_copy)
            up_thres = gt_sum * 2
            low_thres = gt_sum // 2

            if pred_sum < up_thres and pred_sum > low_thres:
                matched_dict[seg_labeled[x, y, z]] = True
            else:
                fp += 1
        else:
            fp += 1

    for seg_ob_id in np.unique(seg_labeled):
        if seg_ob_id != 0 and not matched_dict[seg_ob_id]:
            fn += 1
        elif seg_ob_id != 0 and matched_dict[seg_ob_id]:
            tp += 1

    return tp, fp, fn
