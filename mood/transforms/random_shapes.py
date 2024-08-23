import math
import random
from typing import List, Optional, Tuple

import numpy as np
import skimage


class RandomShapeGenerator:
    """
    An object to generate a 3D mask with random shapes inside.

    Given the scale of the shape (i.e. its approximate size),
    the generator will first perform a random walk inside a
    cube of that scale to select some points. From these points,
    the generator will create a convex shape. The cube containing
    the shape will eventually be inserted randomly inside the image.

    Multiple shapes of different scales can be generated in the
    same mask.

    An input mask can be passed in order to limit where the shapes
    can be generated.
    """

    def __init__(self, radius: Optional[int] = None, n_points: int = 100) -> None:
        """
        Parameters
        ----------
        radius : Optional[int] (optional, default=None)
            The spatial step of the random walk used to generate a shape.
            If not passed, the spatial step will depend on the wanted sizes
            of the shapes.
        n_points : int (optional, default=100)
            The number of steps of the random walk used to generate a shape.
        """
        self.radius = radius
        self.n_points = n_points

    @staticmethod
    def show_in_mask(anomalies: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Draws the anomalies on a brain mask.

        Parameters
        ----------
        anomalies : np.ndarray
            The anomalies generated with `get_random_shapes`.
        mask : np.ndarray
            The mask of the brain (1s and 0s).

        Returns
        -------
        np.ndarray
            The mask with anomalies (dark volumes inside the mask).
        """
        mask = mask.astype(np.int16)
        dark_anomalies = (~anomalies.astype(bool)).astype(np.int16)

        return mask * dark_anomalies

    def get_random_shapes(
        self,
        img: np.ndarray,
        shape_proportions: List[float],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generates random shapes in an empty image, whose size is the same as that of the input image.

        The approximate size of the shapes is computed thanks to shape_proportions.
        shape_proportions must contain floats between 0 and 1. shape_proportion represents
        the relative size of the shape compared to the size of the image
        (i.e. shape_size = shape_proportion*image_size).

        N shapes are generated where N is the length of shape_proportions.

        A mask can be passed in order to generate shapes inside a certain region of the image.

        Parameters
        ----------
        img : np.ndarray
            The input image. The image must be 3D and cubic.
        shape_proportions : List[float]
            Relative sizes of shapes compared to the size of the image.
            Must be between 0 and 1.
            One shape per shape_proportion will be generated.
        mask : Optional[np.ndarray] (optional, default=None)
            A mask to select the center of shapes inside a certain region.
            mask's size must be the same as that of the image.

        Returns
        -------
        np.ndarray
            An image with random shapes (pixels equal to 1) inside a background (0s).
            The image has the same sizes as that of the input.
        """
        img_size = self._get_size(img)
        n_shapes = len(shape_proportions)
        shape_scales = self._get_scales(shape_proportions, img_size)
        radii = self._get_radii(shape_scales)
        mask = self._get_mask(img, mask)
        individual_shapes = np.array([np.zeros_like(img) for _ in range(n_shapes)])

        for k in range(n_shapes):
            scale = shape_scales[k]
            r = radii[k]
            msk = self._draw_border_mask(img_size, margin=scale // 2) * mask
            shape_center = self._get_shape_center(msk)
            shape = self._draw_shape(scale, r)
            individual_shapes[
                k,
                shape_center[0] - scale // 2 : shape_center[0] + (scale + 1) // 2,
                shape_center[1] - scale // 2 : shape_center[1] + (scale + 1) // 2,
                shape_center[2] - scale // 2 : shape_center[2] + (scale + 1) // 2,
            ] = shape
        shapes = np.sum(individual_shapes, axis=0)
        shapes = (shapes > 0).astype(np.int16)

        return shapes * mask

    @staticmethod
    def _get_size(img: np.ndarray) -> int:
        """
        Gets the size of a 3D cubic image.

        Parameters
        ----------
        img : np.ndarray
            A 3D image.

        Returns
        -------
        int
            The size of the image.

        Raises
        ______
        AssertionError
            If the image is not a 3D cubic image.
        """
        img_shape = img.shape
        assert len(img_shape) == 3, "Image must be 3D."
        assert (img_shape[0] == img_shape[1]) and (
            img_shape[0] == img_shape[2]
        ), "Image must be cubic."

        return img_shape[0]

    @staticmethod
    def _get_scales(shape_proportions: List[int], img_size: int) -> np.ndarray:
        """
        Returns shape scales (in pixel) from shape proportions.

        Parameters
        ----------
        shape_proportions : List[int]
            The proportions of the shape relative to the image.
        img_size : int
            The size of the image.

        Returns
        -------
        np.ndarray
            The scales of the shapes.

        Raises
        ------
        AssertionError
            If the proportions are not between 0 and 1.
        """
        assert (0 < np.array(shape_proportions)).all() and (
            np.array(shape_proportions) <= 1
        ).all(), "Shape proportions must be between 0 an 1."
        shape_scales = (np.array(shape_proportions) * img_size).astype(np.int16)

        return shape_scales

    def _get_radii(self, shape_scales: List[int]) -> List[int]:
        """
        Returns list of radii to generate shapes

        If the user has passed a radius value, it will be used for
        all shapes. Else, the radius depends on the size of the
        shape.

        Parameters
        ----------
        shape_scales : List[int]
            The approximate size of the shapes.

        Returns
        -------
        List[int]
            The list of radii.
        """
        if self.radius:
            radius = [int(self.radius) for _ in range(len(shape_scales))]
        else:
            radius = (shape_scales // 10).astype(np.int16)

        return radius

    @staticmethod
    def _get_mask(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Checks the mask, if passed by the user. Otherwise use the image
        to create a mask from non-zero pixels.

        Parameters
        ----------
        img : np.ndarray
            The input image.
        mask : Optional[np.ndarray] (optional, default=None)
            The mask passed by the user.

        Returns
        -------
        np.ndarray
            A mask of the same shape as the image.

        Raises
        ------
        ValueError
            If the mask is not of the same shape as the image.
        """
        if mask is not None:
            img_size = img.shape[0]
            if mask.shape != (img_size, img_size, img_size):
                raise ValueError("Mask should have the same size as that of the image.")
            mask = mask.astype(np.int16)
        elif not mask:
            mask = (img > 0).astype(np.int16)

        return mask

    def _draw_shape(self, shape_scale: int, radius: int) -> np.ndarray:
        """
        Draws a single shape inside a background image.

        Parameters
        ----------
        shape_scale : int
            The size of the background image (which is approximately the size of the shape).
        radius : int
            The radius used to generate the shape.

        Returns
        -------
        np.ndarray
            An 3D image of size equals to `shape_scale` with a shape inside (1s, vs 0s for background).
        """
        img = np.zeros((shape_scale, shape_scale, shape_scale))
        initial_center = [shape_scale // 2, shape_scale // 2, shape_scale // 2]
        X, Y, Z = self._random_walk(self.n_points, initial_center, radius, shape_scale)
        img[X, Y, Z] = 1
        img = self._draw_shape_from_points(img, radius)

        return img

    @classmethod
    def _random_walk(
        cls,
        n_points: int,
        initial_point: Tuple[int, int, int],
        radius: int,
        img_size: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Gets a random list of points thanks to a 3D random walk with a fixed step.

        `n_points` steps will be executed, with a fixed spatial step
        equal to `radius`. At each step, it will be checked that the
        point is at least `radius` pixels away from the border of the
        image.

        Parameters
        ----------
        n_points : int
            The number of steps performed.
        initial_center : Tuple[int, int, int]
            x, y and z coordinates of the initial point.
        radius : int
            The spatial step of the random walk.
        img_size : int
            The size of the image.
            Used to check that the points are at least `radius` pixels away
            from the border of the image.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            x, y and z coordinates of the points.
        """
        X = [initial_point[0]]
        Y = [initial_point[1]]
        Z = [initial_point[2]]
        center = initial_point
        cnt = 0
        while cnt < n_points:
            candidate_center = cls._get_point_on_sphere(center, radius)
            if cls._ball_fits(candidate_center, radius, img_size):
                center = candidate_center
                X.append(center[0])
                Y.append(center[1])
                Z.append(center[2])
                cnt += 1

        return X, Y, Z

    @staticmethod
    def _get_point_on_sphere(
        center: Tuple[int, int, int], radius: int
    ) -> Tuple[int, int, int]:
        """
        Gets a random point on a sphere.

        Parameters
        ----------
        center : Tuple[int, int, int]
            x, y and z coordinates of the center of the sphere.
        radius : int
            Radius of the sphere.

        Returns
        -------
        Tuple[int, int, int]
            x, y and z coordinates of the point on the sphere.
        """
        theta = 2 * math.pi * random.random()
        phi = math.pi * random.random()

        x = round(radius * math.cos(theta) * math.sin(phi) + center[0])
        y = round(radius * math.sin(theta) * math.sin(phi) + center[1])
        z = round(radius * math.cos(phi) + center[2])

        return x, y, z

    @staticmethod
    def _ball_fits(center: Tuple[int, int, int], radius: int, img_size: int) -> bool:
        """
        Checks that a ball fits inside a 3D cubic image.

        Parameters
        ----------
        center : Tuple[int, int, int]
            x, y and z coordinates of the center of the ball.
        radius : int
            The radius of the ball.
        img_size : int
            The size of the image.

        Returns
        -------
        bool
            Whether the ball fits in the image.
        """
        x_fits = radius <= center[0] < img_size - radius
        y_fits = radius <= center[1] < img_size - radius
        z_fits = radius <= center[2] < img_size - radius

        return x_fits and y_fits and z_fits

    @staticmethod
    def _draw_shape_from_points(img: np.ndarray, radius: int) -> np.ndarray:
        """
        Adds a shape on an image.

        The input image is supposed to have a background
        equal to 0 and some foreground points, from which
        a convex shape will be created.

        Parameters
        ----------
        img : np.ndarray
            The input image.
        radius : int
            The radius of the ball used for dilation and opening.

        Returns
        -------
        np.ndarray
            The input image with a convex shape inside.
        """
        img = skimage.morphology.dilation(
            img, footprint=skimage.morphology.ball(radius)
        )
        img = skimage.morphology.convex_hull_image(img)
        img = skimage.morphology.opening(img, footprint=skimage.morphology.ball(radius))

        return img

    @staticmethod
    def _get_shape_center(mask: np.ndarray) -> Tuple[int, int, int]:
        """
        Gets a random point inside a mask.

        Parameters
        ----------
        mask : np.ndarray
            The mask with 0s and 1s or Falses and Trues.

        Returns
        -------
        Tuple[int, int, int]
            x, y and z coordinates of the points.
        """
        weights = mask.astype(np.int16)
        normalized = weights.ravel() / float(weights.sum())
        indices = np.random.choice(mask.size, replace=False, p=normalized)
        idx, idy, idz = np.unravel_index(indices, mask.shape)

        return idx, idy, idz

    @staticmethod
    def _draw_border_mask(img_size: int, margin: int) -> np.ndarray:
        """
        Creates a 3D mask removing the border.

        Parameters
        ----------
        img_size : int
            The size of the mask that will be created.
        margin : int
            The number of pixels to exclude next to the border.

        Returns
        -------
        np.ndarray
            The mask with 0s and 1s.
        """
        mask = np.zeros((img_size, img_size, img_size))
        mask[margin:-margin, margin:-margin, margin:-margin] = 1

        return mask
