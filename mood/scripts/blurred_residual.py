import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.filters import gaussian
from tqdm import tqdm

caps_dir = Path("/root_dir/maps/MAPS_MS_BetaVAE_0/split-4/best-loss/CapsOutput/")
tsv_path = Path("/root_dir/data/brain/caps_test/pixel_level.tsv")


df = pd.read_csv(tsv_path, sep="\t")
list_files = df[['participant_id', 'session_id']].agg('_'.join, axis=1).to_list()
with tqdm(total=70) as pbar:
    for root, _, files in os.walk(caps_dir):
        for f in files:
            if "_input.nii.gz" in f:
                subject_session = f.split("_input")[0]
                if subject_session in list_files:
                    pbar.update(1)
                    img = nib.load(caps_dir / root / f).get_fdata(dtype="float32")
                    blurred = gaussian(img, sigma=1)
                    blurred[img == 0] = 0.0
                    out = nib.load(caps_dir / root / f.replace("input", "output")).get_fdata(dtype="float32")
                    res = blurred - out

                    nifti = nib.Nifti1Image(res.astype(np.float32), np.eye(4))
                    nib.save(nifti, caps_dir / root / f.replace("input", "residual_blurred"))
