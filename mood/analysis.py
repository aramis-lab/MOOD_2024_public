#%%
from mood.transforms.utils import save_image, extract_image
from mood.metrics.utils import get_image_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from skimage.transform import resize
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, convex_hull_image
from skimage.filters import sobel, median

#%%
maps = "GANs/pix2pix/attn_unet_sobel_aug_100/"
split = 4
suffix = '_1'

root_maps = Path(f"/root_dir/maps/{maps}/split-{split}/best-loss/")
tsv_path = root_maps / f"test_pixel/detailed_results{suffix}.tsv"
df = pd.read_csv(tsv_path, sep="\t")
anomaly = pd.read_csv("/root_dir/data/brain/caps_test/subjects_sessions.tsv", sep="\t")
df = df.merge(anomaly, how="left", on=["participant_id", "session_id"])
df

#%%
mean_val = extract_image(f"/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100/split-4/mean_val_res{suffix}.nii.gz")
std_val = extract_image(f"/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100/split-4/std_val_res{suffix}.nii.gz")
std_val = median(std_val, footprint=np.ones((3,3,3)))

#%%
def post_processing(input_img: np.ndarray, output_img: np.ndarray, threshold: float, size_thres: float) -> np.ndarray:
    """
    Here is the postprocessing applies to the outputs to have predictions.
    """
    output_img = resize(output_img, output_shape=(256,256,256))
    mask = binary_closing((input_img > 0), footprint=np.ones((5,5,5)))    
    output_img = output_img * mask

    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

    residual = input_img - output_img
    proba = (np.abs(residual)-mean_val) / std_val

    edges = (sobel(input_img) > 0.1)
    inside = ~edges.astype(bool)
    proba = proba * inside

    pred = proba > threshold
    pred = binary_closing(pred, footprint=np.ones((5,5,5)))
    pred = binary_opening(pred, footprint=np.ones((5,5,5)))

    pred = remove_small_objects(pred, min_size=size_thres, connectivity=2)

    pred = convex_hull_image(pred)

    return pred.astype(np.int8)

#%%
subject = "675"
session = "M003"
input_img = extract_image(f"/root_dir/data/brain/caps_test/subjects/sub-{subject}/ses-{session}/custom/sub-{subject}_ses-{session}_mood.nii.gz")
mask = extract_image(f"/root_dir/data/brain/caps_test/subjects/sub-{subject}/ses-{session}/custom/sub-{subject}_ses-{session}_mask.nii.gz")
output_img = extract_image(root_maps / f"CapsOutput/subjects/sub-{subject}/ses-{session}/custom/sub-{subject}_ses-{session}_output{suffix}.nii.gz")

#%%
size_thres = 600
threshold = 2.5

pred = post_processing(input_img, output_img, threshold=threshold, size_thres=size_thres)
tps, fps, fns = get_image_confusion_matrix(pred, mask, size_thres=size_thres)
print(f"TPs: {tps} | FPs: {fps} | FNs: {fns}")
save_image(pred, "../pred.nii.gz")

# %%
