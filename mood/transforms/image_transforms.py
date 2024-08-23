import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "RemoveSlice",
    "CorruptSlice",
    "GlobalBlurring",
    "GlobalElasticDeformation",
    "Ghosting",
    "Spike",
    "BiasField",
    "Noise",
    "Compose",
    "RandomCompose",
]


class GlobalTransform(ABC):
    "Template for global transforms."

    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        pass


class SliceTransform(GlobalTransform, ABC):
    """Template for transforms on slices."""

    def __init__(
        self,
        n_slices: Union[int, Tuple[int, int]] = (1, 5),
    ) -> None:
        """
        Parameters
        ----------
        n_slices : Union[int, Tuple[int, int]] (optional, default=(1, 5))
            Number of slices to remove.
            If tuple, the lower and upper bound of the uniform
            discrete distribution from which the parameter will be
            sampled.
        """
        try:
            assert isinstance(n_slices, int)
        except AssertionError:
            assert isinstance(
                n_slices, Iterable
            ), "n_slices must be an int or an iterable."
            assert len(n_slices) == 2, "n_slices must contain two elements."
        self.n_slices = n_slices

    @staticmethod
    def _is_null_slice(img: np.ndarray, dim: int, slice_idx: int) -> bool:
        """
        Checks whether a slice is trivial (i.e. with less than
        5% non-zero values).

        Parameters
        ----------
        img : np.ndarray
            The image.
        dim : int
            The dimension along which the slice will be taken (0, 1 or 2).
        slice_idx : int
            The index of the slice.

        Returns
        -------
        bool
            Whether it is a trivial slide.
        """
        if dim == 0:
            s = img[slice_idx, :, :]
        elif dim == 1:
            s = img[:, slice_idx, :]
        else:
            s = img[:, :, slice_idx]
        non_null_prop = np.count_nonzero(s) / s.size

        return non_null_prop < 0.05

    @abstractmethod
    def _modify_slice(self, img: np.ndarray, dim: int, slice_idx: int) -> np.ndarray:
        """
        Modifies the slice in the image.

        Parameters
        ----------
        img : np.ndarray
            The image.
        dim : int
            The dimension along which the slice will be taken (0, 1 or 2).
        slice_idx : int
            The index of the slice.

        Returns
        -------
        np.ndarray
            The modified image.
        """
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """

        img_shape = img.shape
        if isinstance(self.n_slices, int):
            n_slices = self.n_slices
        else:
            n_slices = random.randint(self.n_slices[0], self.n_slices[1])

            dim = np.random.choice(3, size=n_slices, replace=True)
            for d in dim:
                trivial_slice = True
                while trivial_slice:
                    slice_idx = np.random.choice(img_shape[d])
                    trivial_slice = self._is_null_slice(img, d, slice_idx)

                img = self._modify_slice(img, d, slice_idx)

        return img


class RemoveSlice(SliceTransform):
    """Randomly remove slices in the image."""

    @staticmethod
    def _modify_slice(img: np.ndarray, dim: int, slice_idx: int) -> np.ndarray:
        """
        Removes the slice in the image.

        Parameters
        ----------
        img : np.ndarray
            The image.
        dim : int
            The dimension along which the slice will be taken (0, 1 or 2).
        slice_idx : int
            The index of the slice.

        Returns
        -------
        np.ndarray
            The modified image.
        """
        if dim == 0:
            img[slice_idx, :, :] = 0.0
        elif dim == 1:
            img[:, slice_idx, :] = 0.0
        else:
            img[:, :, slice_idx] = 0.0
        return img


class CorruptSlice(SliceTransform):
    """Randomly corrupt slices in the image by darkening or lightening."""

    def __init__(
        self,
        corruption: Union[float, Tuple[float, float]] = (0.1, 2.0),
        n_slices: Union[int, Tuple[int, int]] = (1, 5),
    ) -> None:
        """
        Parameters
        ----------
        corruption : Union[float, Tuple[float, float]] (optional, default=(0.1, 2.0))
            The increase (> 1.0) or dicrease (< 1.0) of brightness in a corrupted slice.
            If tuple, the lower and upper bound of the uniform
            distribution from which the parameter will be
            sampled (a zone around 1.0 is excluded to have an actual corruption).
        n_slices : Union[int, Tuple[int, int]] (optional, default=(1, 5))
            Number of slices to remove.
            If tuple, the lower and upper bound of the uniform
            discrete distribution from which the parameter will be
            sampled.

        Raises
        ______
        AssertionError
            If corruption_range is not ordered or is too close to 1.
        """
        super().__init__(n_slices)
        self.margin = 0.1
        try:
            assert isinstance(corruption, float)
        except AssertionError:
            assert isinstance(
                corruption, Iterable
            ), "corruption must be a float or an iterable."
            assert len(corruption) == 2, "corruption must contain two elements."
            assert (corruption[0] < 1.0 - self.margin) or (
                1.0 + self.margin < corruption[1]
            ), "Corruption interval is too close to 1."
        else:
            assert (corruption < 1.0 - self.margin) or (
                1.0 + self.margin < corruption
            ), "corruption is too close to 1.0."
        self.corruption = corruption

    def _find_corruption_factor(self) -> float:
        """
        Samples a corruption factor.

        Returns
        -------
        float
            The corruption factor.
        """
        if isinstance(self.corruption, float):
            corruption_factor = self.corruption
        else:
            corruption_too_small = True
            while corruption_too_small:
                corruption_factor = random.uniform(
                    self.corruption[0], self.corruption[1]
                )
                corruption_too_small = (
                    1.0 - self.margin <= corruption_factor <= 1.0 + self.margin
                )

        return corruption_factor

    def _modify_slice(self, img: np.ndarray, dim: int, slice_idx: int) -> np.ndarray:
        """
        Darkens or lightnens a slice.

        Parameters
        ----------
        img : np.ndarray
            The image.
        dim : int
            The dimension along which the slice will be taken (0, 1 or 2).
        slice_idx : int
            The index of the slice.

        Returns
        -------
        np.ndarray
            The modified image.
        """
        corruption = self._find_corruption_factor()

        if dim == 0:
            img[slice_idx, :, :] = img[slice_idx, :, :] * corruption
        elif dim == 1:
            img[:, slice_idx, :] = img[:, slice_idx, :] * corruption
        else:
            img[:, :, slice_idx] = img[:, :, slice_idx] * corruption

        return img


class GlobalBlurring(GlobalTransform):
    """Blurs an image with a Gaussian filter."""

    def __init__(
        self,
        std: Union[float, Tuple[float, float]] = (1.0, 2.0),
    ) -> None:
        """
        Parameters
        ----------
        std : float (optional, default=(1.0, 2.0))
            Standard deviation of the Gaussian kernel.
            If tuple, the lower and upper bound of the uniform
            distribution from which the parameter will be
            sampled.
        """
        self.std = std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        from torchio.transforms import RandomBlur

        transform = RandomBlur(std=self.std)
        img = transform(img[None, :, :, :]).squeeze(0)

        return img


class GlobalElasticDeformation(GlobalTransform):
    """Applies Elastic Deformation to the whole image."""

    def __init__(
        self,
        num_control_points: Union[int, Tuple[int, int, int]] = 30,
        max_displacement: Union[float, Tuple[float, float, float]] = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        num_control_points : Union[int, Tuple[int, int, int]] (optional, default=30.0)
            Number of control points along each dimension of the coarse grid.
        max_displacement : Union[float, Tuple[float, float, float]] (optional, default=10.0)
            Maximum displacement along each dimension at each control point.
        """
        self.num_control_points = num_control_points
        self.max_displacement = max_displacement

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        from torchio.transforms import RandomElasticDeformation

        transform = RandomElasticDeformation(
            num_control_points=self.num_control_points,
            max_displacement=self.max_displacement,
        )
        img = transform(img[None, :, :, :]).squeeze(0)

        return img


class Ghosting(GlobalTransform):
    """Applies MRI ghosting artifact to the image."""

    def __init__(
        self,
        intensity: Union[float, Tuple[float, float]] = (0.5, 1),
    ) -> None:
        """
        Parameters
        ----------
        intensity : Union[float, Tuple[float, float]] (optional, default=(0.5, 1))
            Positive number representing the artifact strength
            with respect to the maximum of the k-space.
            If tuple, the lower and upper bound of the uniform
            distribution from which the parameter will be
            sampled.
        """
        self.intensity = intensity

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        from torchio.transforms import RandomGhosting

        transform = RandomGhosting(intensity=self.intensity)
        img = transform(img[None, :, :, :]).squeeze(0)

        return img


class Spike(GlobalTransform):
    """Applies MRI spike artifact to the image."""

    def __init__(
        self,
        intensity: Union[float, Tuple[float, float]] = (0.5, 1),
    ) -> None:
        """
        Parameters
        ----------
        intensity : Union[float, Tuple[float, float]] (optional, default=(0.5, 1))
            Positive number representing the artifact strength
            with respect to the maximum of the k-space.
            If tuple, the lower and upper bound of the uniform
            distribution from which the parameter will be
            sampled.
        """
        self.intensity = intensity

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        from torchio.transforms import RandomSpike

        transform = RandomSpike(intensity=self.intensity)
        img = transform(img[None, :, :, :]).squeeze(0)

        return img


class BiasField(GlobalTransform):
    """Applies MRI bias field artifact to the image."""

    def __init__(
        self,
        coefficients: Union[float, Tuple[float, float]] = (0.5, 2.0),
    ) -> None:
        """
        Parameters
        ----------
        coefficients : Union[float, Tuple[float, float]] (optional, default=(0.5, 2.0))
            Maximum magnitude of polynomial coefficients.
            If tuple, the lower and upper bound of the uniform
            distribution from which the parameter will be
            sampled.
        """
        try:
            assert isinstance(coefficients, float)
        except AssertionError:
            assert isinstance(
                coefficients, Iterable
            ), "coefficients must be a float or an iterable."
            assert len(coefficients) == 2, "coefficients must contain two elements."
        self.coefficients = coefficients

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        from torchio.transforms import RandomBiasField

        if isinstance(self.coefficients, float):
            coefficients = self.coefficients
        else:
            coefficients = random.uniform(self.coefficients[0], self.coefficients[1])

        transform = RandomBiasField(coefficients=coefficients)
        img = transform(img[None, :, :, :]).squeeze(0)

        return img


class Noise(GlobalTransform):
    """Applies noise to the image."""

    def __init__(
        self,
        std: Union[float, Tuple[float, float]] = (0.05, 0.2),
    ) -> None:
        """
        Parameters
        ----------
        std : Union[float, Tuple[float, float]] (optional, default=(0.05, 0.2))
            Standard deviation of the Gaussian noise.
            If tuple, lower and upper bounds of the uniform
            distribution from which the parameter will be
            sampled.
        """
        self.std = std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        from torchio.transforms import RandomNoise

        transform = RandomNoise(std=self.std)
        img = transform(img[None, :, :, :]).squeeze(0)

        return img


class Compose(GlobalTransform):
    """To compose transforms."""

    def __init__(self, transforms: List[GlobalTransform]):
        """
        Parameters
        ----------
        transforms : List[GlobalTransform]
            List of transforms to apply.
        """
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the transform to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        for transform in self.transforms:
            img = transform(img)

        return img


class RandomCompose(GlobalTransform):
    """To randomly compose transforms."""

    def __init__(
        self,
        transforms: List[GlobalTransform],
        n_transforms: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        """
        Parameters
        ----------
        transforms : List[GlobalTransform]
            List of transforms to apply.
        n_transforms : Optional[Union[int, Tuple[int, int]]] (optional, default=None)
            Number of transforms to compose.
            If tuple, the lower and upper bound of the uniform
            discrete distribution from which the parameter will be
            sampled.
            If None, set to (1, len(transforms)).
        """
        self.transforms = transforms
        if n_transforms is None:
            self.n_transforms = (1, len(transforms))
        else:
            try:
                assert isinstance(n_transforms, int)
            except AssertionError:
                assert isinstance(
                    n_transforms, Iterable
                ), "n_slices must be an int or an iterable."
                assert len(n_transforms) == 2, "n_slices must contain two elements."
            self.n_transforms = n_transforms

    def _select_transforms(self) -> List[GlobalTransform]:
        """
        Randomly selects transforms.

        Returns
        -------
        List[GlobalTransform]
            The selected transforms.
        """
        if isinstance(self.n_transforms, int):
            n_transforms = self.n_transforms
        else:
            n_transforms = random.randint(self.n_transforms[0], self.n_transforms[1])
        idx = np.random.choice(len(self.transforms), size=n_transforms, replace=False)
        transforms = np.array(self.transforms)[idx]

        return transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly selects transforms and applies them to an image.

        Parameters
        ----------
        img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The transformed image.
        """
        transforms = self._select_transforms()
        for transform in transforms:
            img = transform(img)

        return img
