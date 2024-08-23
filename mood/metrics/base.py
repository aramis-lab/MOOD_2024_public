from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

class Metric(ABC):
    """Base class for metrics."""

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the cumulative arrays.
        """

    @abstractmethod
    def aggregate(self) -> Union[float, Dict[str, float]]:
        """
        Computes final metric.

        Returns
        -------
        float
            The aggregated metric.
        """

    @abstractmethod
    def append(self, y_pred: Union[float, np.ndarray], y: Union[int, np.ndarray]) -> None:
        """
        Computes useful values from groundtruth and predictions
        and adds them values to cumulative arrays.

        Parameters
        ----------
        y_pred : np.ndarray or float
            The predictions.
        y : np.ndarray or int
            The ground truth.
        """
