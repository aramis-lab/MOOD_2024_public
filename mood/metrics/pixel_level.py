from functools import partial
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from mood.metrics.utils import get_image_confusion_matrix
from mood.post_processing.process_residual_pixel import PostProcessor

from mood.metrics.base import Metric


class PixelF1Score(Metric):
    def __init__(
        self, min_size: Optional[int] = 5 * 5 * 5, n_proc: int = 2
    ):
        """
        Parameters
        ----------
        min_size : Optional[int] (optional, default=125)
            Threshold to discard small predictions.
        n_proc : int (optional, default=2)
            Number of processes to parallelize metric computation across a batch.
        """
        self.n_proc = n_proc
        self.min_size = min_size
        self.reset()

    def reset(self) -> None:
        """
        Resets the cumulative scores.
        """
        self.results_per_img = []

    def aggregate(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Computes final F1 score based on cumulated TPs, FPs and FNs.

        Returns
        -------
        Dict[str, float]
            The aggregated scores..
        """
        df = pd.DataFrame(self.results_per_img)
        tps = df["TP"].sum()
        fps = df["FP"].sum()
        fns = df["FN"].sum()
        f1_score = 2 * tps / (2 * tps + fps + fns)
        matrix = {
            "F1-score": f1_score,
            "TP": float(tps),
            "FP": float(fps),
            "FN": float(fns),
        }
        return matrix, df

    def append(self, y_pred: np.ndarray, y: np.ndarray, infos: Iterable[Dict[str, str]]) -> None:
        """
        Computes TPs, FPs and FNs in an image and adds these
        values to cumulative scores.

        Parameters
        ----------
        y_pred : np.ndarray
            The predictions.
        y : np.ndarray
            The ground truth.
        infos : Iterable[Dict[str, str]]
            Information on the images (e.g. type of anomaly).
        """
        infos = [dict(zip(infos, t)) for t in zip(*infos.values())]
        multi_pool = mp.Pool(processes=self.n_proc)
        outputs = multi_pool.starmap(
            partial(
                get_image_confusion_matrix,
                size_thres=self.min_size,
            ),
            zip(y_pred, y),
        )
        multi_pool.close()
        multi_pool.join()
        for res, info in zip(outputs, infos):
            info["TP"] = res[0]
            info["FP"] = res[1]
            info["FN"] = res[2]
            self.results_per_img.append(info)


class ThresholdFinder(Metric):
    def __init__(
        self,
        process_fct: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        n_points: int = 10,
        min_size: Optional[int] = 5 * 5 * 5,
        n_proc: int = 2,
    ):
        """
        Parameters
        ----------
        process_fct: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
            The postprocessing function.
        threshold_range : Tuple[float, float] (optional, default=(0.1, 0.9))
            Lower and upper bound of threshold search space.
        n_points : int (optional, default=10)
            Number of thresholds to sample in the search space.
        min_size : Optional[int] (optional, default=125)
            Threshold to discard small predictions.
        n_proc : int (optional, default=2)
            Number of processes to parallelize metric computation across a batch.
        """
        self.n_proc = n_proc
        self.min_size = min_size
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        self.process_fct = process_fct
        self.reset()

    def reset(self) -> None:
        """
        Resets the cumulative scores.
        """
        self.score_storage = {
            thresh: PixelF1Score(
                min_size=self.min_size, n_proc=self.n_proc
            )
            for thresh in self.thresholds
        }

    def aggregate(self) -> Tuple[float, Dict[str, float]]:
        """
        Computes final F1 scores for each threshold, based on cumulated TPs, FPs and FNs.
        """
        outputs = {
            thresh: scores.aggregate()[0] for thresh, scores in self.score_storage.items()
        }
        best_thresh = max(outputs, key=lambda th: outputs.get(th)["F1-score"])

        return best_thresh, outputs

    def append(self, inputs: np.ndarray, outputs: np.ndarray, mask: np.ndarray, infos: Iterable[Dict[str, str]]) -> None:
        """
        For each threshold, computes TPs, FPs and FNs in an image and adds these
        values to cumulative scores.
        """
        for threshold, scores in self.score_storage.items():
            post_processor = PostProcessor(process_fct=self.process_fct, threshold=threshold, n_proc=self.n_proc)
            preds, _ = post_processor.process(inputs, outputs)
            scores.append(preds, mask, infos)
