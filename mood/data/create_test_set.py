import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

import mood.transforms.image_transforms as image
import mood.transforms.local_transforms as local
from mood.transforms.utils import extract_image, save_image, set_randomness


##### params #####
caps_path = "/root_dir/data/abdominal/caps_abdom_custom"
tsv_path = "/root_dir/data/abdominal/caps_abdom/split/test_baseline.tsv"
n_per_transform = 2
n_compose = 4
seed = 42
##################

image_transforms = {
    "CorruptSlice": image.CorruptSlice(
        corruption=(0.0, 3.0), n_slices=(1, 10),
    ),
    "GlobalBlurring": image.GlobalBlurring(std=(0.5, 2.0)),
    "GlobalElasticDeformation": image.GlobalElasticDeformation(
        num_control_points=20, max_displacement=10,
    ),
    "Ghosting": image.Ghosting(intensity=(0.5, 1.0)),
    "Spike": image.Spike(intensity=(0.5, 1.5)),
    "BiasField": image.BiasField(coefficients=(0.5, 3.0)),
    "Noise": image.Noise(std=(0.01, 0.1)),
}
image_transforms["ImageCompose"] = image.RandomCompose(
    transforms=[t for _, t in image_transforms.items()], n_transforms=(2, 3)
)

local_transforms = {
    "LocalPixelShuffling": local.LocalPixelShuffling(
        shuffle_factor=(0.5, 1.0),
        n_anomalies=(1, 4),
        anomaly_proportion=(0.1, 0.2),
    ),
    "DarkOrBright": local.DarkOrBright(
        brightness_increase=(0.0, 2.0),
        n_anomalies=(1, 4),
        anomaly_proportion=(0.1, 0.2),
    ),
    "LocalBlurring": local.LocalBlurring(
        std=(1.0, 6.0), n_anomalies=(1, 3), anomaly_proportion=(0.2, 0.3)
    ),
    "LocalElasticDeformation": local.LocalElasticDeformation(
        max_displacement=20,
        n_anomalies=(1, 2),
        anomaly_proportion=(0.3, 0.4),
    ),
}
local_transforms["LocalCompose"] = local.RandomCompose(
    transforms=[t for _, t in local_transforms.items()],
    n_transforms=(2, 3),
    n_anomalies=(1, 3),
    anomaly_proportion=(0.1, 0.4),
)

##### compute transforms #####

caps_path = Path(caps_path)
output_df = pd.read_csv(tsv_path, sep="\t")
subject_list = output_df["participant_id"].to_list()
file_list = [caps_path / "subjects" / s for s in subject_list]
output_df["image level"] = np.nan
output_df["type"] = np.nan
n_times_used = np.zeros_like(file_list).astype(np.float16)

with set_randomness(seed):
    for transform in tqdm(image_transforms, desc="Image transforms"):

        if transform == "ImageCompose":
            n = n_compose
        else:
            n = n_per_transform

        cnt = 0
        used_for_this_anomaly = []
        while cnt < n:

            probs = np.exp(-n_times_used)/sum(np.exp(-n_times_used))
            img_idx = np.random.choice(len(file_list), p=probs)
            subject = file_list[img_idx]

            if subject.name in used_for_this_anomaly:
                continue
            used_for_this_anomaly.append(subject.name)
            n_times_used[img_idx] += 1
            session = f"ses-M00{int(n_times_used[img_idx])}"

            img_path = subject / "ses-M000" / "custom" / f"{subject.name}_ses-M000_mood.nii.gz"
            img = extract_image(img_path)

            output = image_transforms[transform](img)
            output_path = subject / session / "custom" / f"{subject.name}_{session}_mood.nii.gz"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(output, output_path)

            row_df = pd.DataFrame([[subject.name, session, "A", True, transform]], columns=output_df.columns)
            output_df = pd.concat([output_df, row_df])

            cnt += 1


    for transform in tqdm(local_transforms, desc="Local transforms"):

        if transform == "LocalCompose":
            n = n_compose
        else:
            n = n_per_transform

        cnt = 0
        used_for_this_anomaly = []
        while cnt < n:

            probs = np.exp(-n_times_used)/sum(np.exp(-n_times_used))
            img_idx = np.random.choice(len(file_list), p=probs)
            subject = file_list[img_idx]

            if subject.name in used_for_this_anomaly:
                continue 
            used_for_this_anomaly.append(subject.name)
            n_times_used[img_idx] += 1
            session = f"ses-M00{int(n_times_used[img_idx])}"

            img_path = subject / "ses-M000" / "custom" / f"{subject.name}_ses-M000_mood.nii.gz"
            img = extract_image(img_path)

            output, anomaly_mask = local_transforms[transform](img)
            output_path = subject / session / "custom" / f"{subject.name}_{session}_mood.nii.gz"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(output, output_path)

            mask_path = subject / session / "custom" / f"{subject.name}_{session}_mask.nii.gz"
            save_image(anomaly_mask, mask_path)

            row_df = pd.DataFrame([[subject.name, session, "A", True, transform]], columns=output_df.columns)
            output_df = pd.concat([output_df, row_df])

            cnt += 1

    output_df.to_csv(caps_path / "subjects_sessions.tsv", sep="\t", index=False)
