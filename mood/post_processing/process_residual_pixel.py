import time
from functools import partial
from typing import Callable
import numpy as np
import torch.multiprocessing as mp
from skimage.filters import median, sobel
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_opening,
    convex_hull_image,
    remove_small_objects,
)
from skimage.transform import resize
from mood.transforms.utils import extract_image
# mean_val = extract_image("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100/split-4/mean_val_res_0.nii.gz")
# std_val = extract_image("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100/split-4/std_val_res_0.nii.gz")
# std_val = median(std_val, footprint=np.ones((3,3,3)))



class PostProcessor:
    """
    To parallelize post-processing.

    Parameters
    ----------
    process_fct : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        The post-processing function.
    threshold : float
        The thresholding.
    n_proc : int (optional, default=2)
        To parallelize.
    """

    def __init__(
        self, process_fct: Callable[[np.ndarray, np.ndarray], np.ndarray], threshold: float, n_proc: int = 2
    ):
        self.process_fct = process_fct
        self.n_proc = n_proc
        self.threshold = threshold

    def process(self, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """
        Applies post-processing to a batch.
        """
        multi_pool = mp.Pool(processes=self.n_proc)
        outputs = multi_pool.starmap(partial(self.process_fct, threshold=self.threshold), zip(inputs, outputs))
        multi_pool.close()
        multi_pool.join()

        preds = np.array([output[0] for output in outputs])
        times = [output[1] for output in outputs]

        return preds, times



def post_processing(input_img: np.ndarray, output_img: np.ndarray, mean_np: np.ndarray, std_np: np.ndarray, threshold: float, brain:bool = True) -> np.ndarray:
    """
    Postprocessing applies to the outputs to have predictions.
    """
    start_time = time.time()

    std_np = median(std_np, footprint=np.ones((3,3,3))) #??

    if not brain :
        input_img = resize(input_img, output_shape=(256,256,256))
        #output_img = resize(output_img, output_shape=(256,256,256))
    
    mask = binary_closing((input_img > 0), footprint=np.ones((5,5,5)))    
    output_img = output_img * mask

    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

    residual = input_img - output_img
    # proba = np.abs(residual)
    proba = (np.abs(residual)-mean_np) / std_np

    edges = (sobel(input_img) > 0.1)
    inside = ~edges.astype(bool)
    proba = proba * inside

    pred = proba > threshold

    pred = binary_closing(pred, footprint=np.ones((5,5,5)))
    pred = binary_opening(pred, footprint=np.ones((5,5,5)))

    pred = remove_small_objects(pred, min_size=500, connectivity=2)
    labels, cnt = label(pred, connectivity=2, return_num=True)
    new_img = np.zeros_like(pred)
    for l in range(1, cnt+1):
        img_label = (labels == l)
        cvx_img = convex_hull_image(img_label)
        new_img[cvx_img] = 1
    pred = new_img

    total_time = time.time() - start_time

    return pred.astype(np.int8)
