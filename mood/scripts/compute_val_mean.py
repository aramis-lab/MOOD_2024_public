from skimage.transform import resize
import numpy as np
from pathlib import Path
from mood.transforms.utils import extract_image
import pandas as pd
from tqdm import tqdm
from mood.transforms.utils import save_image

maps = "GANs/pix2pix/attn_unet_sobel_aug_100"
split = 4
output_name = "output_4"

#########################

validation_tsv = pd.read_csv(f"/root_dir/maps/{maps}/groups/validation/split-{split}/data.tsv", sep="\t")
subject = validation_tsv["participant_id"]

mean = np.zeros((128,128,128))
root = Path(f"/root_dir/maps/{maps}/split-{split}/best-loss/CapsOutput/subjects/")
cnt = 0
for s in tqdm(subject):
    output_ = extract_image(root / s / "ses-M000" / "custom" / f"{s}_ses-M000_{output_name}.nii.gz")
    input_ = extract_image(root / s / "ses-M000" / "custom" / f"{s}_ses-M000_input.nii.gz")
    res = np.abs(input_ - output_)
    mean += res
    cnt += 1
mean /= cnt

std = np.zeros((128,128,128))
for s in tqdm(subject):
    output_ = extract_image(root / s / "ses-M000" / "custom" / f"{s}_ses-M000_{output_name}.nii.gz")
    input_ = extract_image(root / s / "ses-M000" / "custom" / f"{s}_ses-M000_input.nii.gz")
    res = np.abs(input_ - output_)
    std += (res-mean)**2
std = np.sqrt(std / 49)

save_image(resize(mean, output_shape=(256,256,256)), f"/root_dir/maps/{maps}/split-{split}/mean_val_res_{output_name}.nii.gz")
save_image(resize(std, output_shape=(256,256,256)), f"/root_dir/maps/{maps}/split-{split}/std_val_res_{output_name}.nii.gz")