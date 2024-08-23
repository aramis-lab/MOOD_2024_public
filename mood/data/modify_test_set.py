from pathlib import Path
from tqdm import tqdm

import mood.transforms.image_transforms as image
import mood.transforms.local_transforms as local
from mood.transforms.utils import extract_image, save_image


##### params #####
caps_path = "data/caps_test"
n_per_transform = 10
n_compose = 20
##################

image_transforms = {    "GlobalElasticDeformation": image.GlobalElasticDeformation(
        num_control_points=5, max_displacement=20,
    )}

##### compute transforms #####

caps_path = Path(caps_path)
subject_list = [f for f in (caps_path / "subjects").iterdir() if ('sub-224' in str(f)) or ('sub-264' in str(f))]
session = "ses-M000"


for transform in tqdm(image_transforms, desc="Image transforms"):

    if transform == "ImageCompose":
        n = n_compose
    else:
        n = n_per_transform

    for subject in subject_list:

        img_path = subject / session / "custom" / f"{subject.name}_{session}_mood.nii.gz"
        img = extract_image(img_path)

        output = image_transforms[transform](img)

        output_path = subject / session / "custom" / f"{subject.name}_{session}_{transform}.nii.gz"
        save_image(output, output_path)
        

# for transform in tqdm(local_transforms, desc="Local transforms"):

#     if transform == "LocalCompose":
#         n = n_compose
#     else:
#         n = n_per_transform

#     for subject in subject_list:

#         img_path = subject / session / "custom" / f"{subject.name}_{session}_mood.nii.gz"
#         img = extract_image(img_path)

#         output, anomaly_mask = local_transforms[transform](img)

#         output_path = subject / session / "custom" / f"{subject.name}_{session}_{transform}.nii.gz"
#         save_image(output, output_path)

#         mask_path = subject / session / "custom" / f"{subject.name}_{session}_desc-{transform}_mask.nii.gz"
#         save_image(anomaly_mask, mask_path)
