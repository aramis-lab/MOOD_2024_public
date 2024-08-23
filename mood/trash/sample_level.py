from typing import Dict, Tuple, Optional

from functools import partial
import numpy as np
import torch.multiprocessing as mp

from sklearn import metrics


class SampleAPScore(Metric):
    def __init__(self, threshold: float, min_size: Optional[int] = 5*5*5, n_proc: int = 2):
        """
        Parameters
        ----------
        threshold : float
            Threshold to binarize the image.
        min_size : Optional[int] (optional, default=None)
            Threshold to discard small predictions.
        n_proc : int (optional, default=2)
            Number of processes to parallelize metric computation across a batch.
        """
        self.n_proc = n_proc
        self.reset()

    def reset(self) -> None:
        """
        Resets the cumulative scores.
        """
        self.label_vals = 0
        self.pred_vals = 0

    def aggregate(self) -> float:
        """
        Computes final AP score.

        Returns
        -------
        float: a single value for AP
        """

        return metrics.average_precision_score(self.label_vals, self.pred_vals)

    def append(self, y_pred: float, y: int) -> None:
        """

        Parameters
        ----------
        y_pred : float
            The predicted image.
        y : int
            The ground truth.
        """

        self.label_vals += y
        self.pred_vals += y_pred
