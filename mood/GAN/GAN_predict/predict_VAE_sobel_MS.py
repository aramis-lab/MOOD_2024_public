import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import skimage.morphology as MM
import torch
import torchvision.transforms as transforms
from skimage.filters import gaussian, median, sobel, threshold_otsu
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, StackDataset


from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage


from mood.GAN.GAN_trainer.model_size_ch16 import AttU_Net as AttU_Net_256_ch16
from mood.GAN.GAN_trainer.model_cd import Generator
from mood.GAN.GAN_trainer.utils import (
    MinMaxNormalization,
    ResidualMean,
    ResizeInterpolation,
    ThresholdNormalization,
)


# Argument parsing

parser = argparse.ArgumentParser(description="prediction model")

parser.add_argument("--caps", help="caps directory", default=None)
parser.add_argument("--json", help="preprocessing json path", default=None)
parser.add_argument("--tsv", help="tsv of sub/sess for inference", default=None)
parser.add_argument("--maps_gen", help="Generator (MAPS) dir", default=None)
parser.add_argument("--vae_split", type=int, help="split number", default=0)
parser.add_argument("--gen_split", type=int, help="split number", default=0)

# added by MS
parser.add_argument(
    "--vae_dir",
    type=str,
    help="VAE model name",
    default="MS_BetaVAE_0",
)
parser.add_argument(
    "--mean_path",
    type=str,
)
parser.add_argument(
    "--median_path",
    type=str,
)
parser.add_argument(
    "--mode",
    type=str,
    default="brain",
)

args = parser.parse_args()

CAPS_DIR = args.caps
LABEL_TSV = args.tsv
PREPROCESSING_JSON = args.json
GEN_PRETRAINED_DIR = args.maps_gen
vae_split = args.vae_split
gen_split = args.gen_split
mode = args.mode
vae_maps_dir = args.vae_dir
input_path = Path(args.mean_path)
median_path = Path(args.median_path)

if mode == "brain":
    affine = np.diag((0.7, 0.7875, 0.7, 0))
# abdom
elif mode == "abdom":
    affine = np.diag((0.72, 0.72, 0.87, 0))
# split = 4
# ROOT_DIR = "/root_dir/"
# CAPS_DIR = f"{ROOT_DIR}/data/brain/caps_brain_custom"
# LABEL_TSV = f"{ROOT_DIR}/data/brain/caps_brain_t1/split/5_fold/split-{split}/validation_baseline.tsv"
# PREPROCESSING_JSON = f"{ROOT_DIR}/misc/MS_extract.json"
# GEN_PRETRAINED_DIR = f"{ROOT_DIR}/maps/GANs/pix2pix/attn_unet_sobel_aug_100"

# mode = 3
# vae_name = "MS_BetaVAE"

# Path for split
split_i_path = os.path.join(GEN_PRETRAINED_DIR, f"split-{gen_split}", "best-loss")

# Path for VAE MAPS
# vae_maps_dir = f"/root_dir/maps/MAPS_{vae_name}_0"

# Load mean image
mean_contour = nib.load(input_path).get_fdata()
mean_tensor = ResizeInterpolation((256))(torch.tensor(mean_contour))
mean_contour = mean_tensor.squeeze().numpy()

# Load median image

median = nib.load(median_path).get_fdata()
median_tensor = ResizeInterpolation((256))(torch.tensor(median))
median = median_tensor.squeeze().numpy()


# Classes & methods


class Sobel(object):

    def __call__(self, image):
        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0)


def voxelwise_ssim_mask(input_img, pseudo_healthy_img, pseudo_healthy_img_type):
    """
    Computes voxelwise SSIM between input_img and pseudo_healthy_img
    and post-processes it appropriately to output binary
    mask to compute hybrid and replace appropriate
    (anomalous) patches in input_img with patches from pseudo_healthy_img

    Args:
        input_img: input image (with potential anomalies)
        pseudo_healthy_img: mean, or VAE output
        pseudo_healthy_img_type: can be either mean or vae_output, useful
            to determine smoothing parameters

    Outputs:
        score: mean SSIM between (anomalous) input image
            (input_img) and pseudo_healthy_img (mean or vae_output)
        prob_inpainting_mask: post-processed voxelwise SSIM
            map between input_img and pseudo_healthy_img
    """

    # Apply appropriate Gaussian filter on input_img

    sigma_input = 1.5
    post_processed_input_img = gaussian(input_img, sigma=sigma_input).astype(np.float32)

    sigma_vae_output = 1.5
    post_processed_pseudo_healthy_img = gaussian(pseudo_healthy_img, sigma=sigma_vae_output)

    # Compute voxelwise SSIM between post_processed_pseudo_healthy_img and input_img
    ssim_score, voxelwise_ssim = structural_similarity(
        post_processed_pseudo_healthy_img,
        post_processed_input_img,
        data_range=1,
        win_size=15,
        gradient=False,
        full=True,
        gaussian_weights=False,
    )

    # Post-process the voxelwise SSIM
    erosion_SSIM = MM.closing(voxelwise_ssim, MM.cube(3))
    prob_inpainting_mask = gaussian(erosion_SSIM, 3)

    return ssim_score, prob_inpainting_mask


def create_hybrid(input_img, vae_output, hybrid_mode, mode):
    """ """

    pseudo_healthy_img = np.copy(vae_output)
    pseudo_healthy_img_type = "vae_output"

    _, prob_inpainting_mask = (
        voxelwise_ssim_mask(
            input_img,
            pseudo_healthy_img,
            pseudo_healthy_img_type=pseudo_healthy_img_type,
        )
    )


    threshold = threshold_otsu(prob_inpainting_mask[prob_inpainting_mask < 0.8])
    bin_inpainting_mask = prob_inpainting_mask < threshold

    inpainted_pseudo_healthy_reconstruction = (
        input_img * (1 - bin_inpainting_mask) + pseudo_healthy_img * bin_inpainting_mask
    )

    return (
        bin_inpainting_mask,
        torch.from_numpy(inpainted_pseudo_healthy_reconstruction)
        .unsqueeze(0)
        .unsqueeze(0)
        .float(),
    )


# Load data

## Transforms

if mode == "brain":
    transforms_input = transforms.Compose(
        [MinMaxNormalization()]
    )

    transforms_output = transforms.Compose(
        [MinMaxNormalization()]
    )
elif mode == "abdom" :
    transforms_input = transforms.Compose(
    [
        MinMaxNormalization(),
        ResizeInterpolation((256, 256, 256)),
        MinMaxNormalization(),
    ]
    )
    transforms_output = transforms.Compose(
        [
            MinMaxNormalization(),
            ResizeInterpolation((256, 256, 256)),
            MinMaxNormalization(),
        ]
    )

# parameters = get_parameters_dict(
#             "custom",
#             "image",
#             False,
#             use_uncropped_image = False,
#             extract_json = "VAE_maps_extract", #"extract_mood",
#             custom_suffix = "*output*", #args.custom_suffix, #"*mood*",
#         )
# DeepLearningPrepareData(
#         caps_directory=Path(
#         f"{vae_maps_dir}/split-{vae_split}/best-loss/CapsOutput"
#     ),
#         tsv_file=Path(LABEL_TSV),
#         n_proc = 1,
#         parameters=parameters,
#     )
t1_weighting_caps_VAE = CapsDatasetImage(
    Path(
        f"{vae_maps_dir}/split-{vae_split}/best-loss/CapsOutput"
    ),  # split & vae_maps_dir added by MS
    Path(LABEL_TSV),
    Path(f"{vae_maps_dir}/tete.json"),
    #Path(f"{vae_maps_dir}/split-{vae_split}/best-loss/CapsOutput/tensor_extraction/VAE_maps_extract.json"),
    train_transformations=transforms_output,
)

t1_weighting_caps_input = CapsDatasetImage(
    Path(CAPS_DIR),
    Path(LABEL_TSV),
    Path(PREPROCESSING_JSON),
    train_transformations=transforms_input,
)

t1_weighting_caps_output = CapsDatasetImage(
    Path(CAPS_DIR),
    Path(LABEL_TSV),
    Path(PREPROCESSING_JSON),
    train_transformations=transforms_output,
)

stacked_dataset = StackDataset(
    t1_weighting_caps_input, t1_weighting_caps_output, t1_weighting_caps_VAE
)

val_loader = DataLoader(
    stacked_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    prefetch_factor=2,
)

# Sort out device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define the generator model

architecture_G = AttU_Net_256_ch16()

optimizer_generator = torch.optim.Adam(
    architecture_G.parameters(), lr=0.0002, betas=(0.9, 0.999)
)

print("1")

scheduler_G = None

Generator_CD1 = Generator(
    architecture_G, optimizer_generator, torch.nn.L1Loss(), scheduler_G
).to(device)

print("2")
Generator_CD1.load(
    Path(
        GEN_PRETRAINED_DIR,
        f"split-{gen_split}",
        "best-loss",
        "gan",
        "generator",
        "modelCD.pth.tar",
    )
)

print("loaded")
for i, (source, target, vae_cop) in enumerate(val_loader):

    # Load data
    print(f"iteration: {i} / {len(val_loader)}")
    participant_id = source["participant_id"][0]
    session_id = source["session_id"][0]
    print(f"{participant_id} {session_id}")
    real_1 = source["image"]
    real_2 = target["image"]
    vae_image = vae_cop["image"]

    # Sort out paths

    session_path = os.path.join(
        split_i_path, "CapsOutput", "subjects", participant_id, session_id
    )

    nifti_path = os.path.join(session_path, "custom")
    os.makedirs(nifti_path, exist_ok=True)
    tensor_path = os.path.join(
        session_path, "deeplearning_prepare_data", "image_based", "custom"
    )
    os.makedirs(tensor_path, exist_ok=True)

    # Load nifti_input
    nifti_input = os.path.join(
        CAPS_DIR,
        "subjects",
        participant_id,
        session_id,
        "custom",
        f"{participant_id}_{session_id}_mood.nii.gz",
    )
    test_nii = nib.load(nifti_input)

    # torch.save(
    #     real_1,
    #     os.path.join(
    #         tensor_path,
    #         f"{participant_id}_{session_id}_image-0_input.pt",
    #     ),
    # )
    # nib.save(
    #     nib.Nifti1Image(
    #         real_1.squeeze().numpy(), header=test_nii.header, affine=test_nii.affine
    #     ),
    #     os.path.join(nifti_path, f"{participant_id}_{session_id}_input.nii.gz"),
    # )

    # Create hybrid between input and VAE reconstruction / mean of inputs
    (
        bin_inpainting_mask,
        real_1,
    ) = create_hybrid(
        real_1.squeeze().numpy(),
        vae_image.squeeze().numpy(),
        hybrid_mode=mode,
        mode = mode,
    )

    print("hybrid created")
    # Save dataframe
    # df_row = [
    #     participant_id,
    #     session_id,
        # pseudo_healthy_img_type,
        # ssim_mean_vae_output,
        # ssim_median_vae_output,
        # ssim_input_pseudo_healthy_reconstruction,
        # ssim_input_mean,
        # ssim_input_median,
    # ]
    # df = pd.concat([pd.DataFrame([df_row], columns=df.columns), df], ignore_index=True)
    # df.to_csv(os.path.join(split_i_path, f"hybrid_logs_{mode}.tsv"), sep="\t", index=False)

    # nib.save(
    #     nib.Nifti1Image(
    #         bin_inpainting_mask.squeeze(),
    #         header=test_nii.header,
    #         affine=test_nii.affine,
    #     ),
    #     os.path.join(
    #         nifti_path,
    #         f"{participant_id}_{session_id}_binary-inpainting-mask_{mode}.nii.gz",
    #     ),
    # )

    # Save hybrid as pt and nii
    # torch.save(
    #     real_1,
    #     os.path.join(
    #         tensor_path,
    #         f"{participant_id}_{session_id}_image-0_hybrid_{mode}.pt",
    #     ),
    # )
    # nib.save(
    #     nib.Nifti1Image(
    #         real_1.squeeze().numpy(), header=test_nii.header, affine=test_nii.affine
    #     ),
    #     os.path.join(
    #         nifti_path,
    #         f"{participant_id}_{session_id}_{vae_maps_dir.name}_hybrid.nii.gz",
    #     ),
    # )

    # Compute sobel and pass it to pre-trained generator

    real_1 = Sobel()(real_1).unsqueeze(0)
    real_1_w = real_1.to(device)
    real_2 = real_2.to(device)

    with torch.no_grad():
        fake_2 = Generator_CD1.forward(real_1_w)
        residual = real_2 - fake_2

    # Save output reconstruction, residual, and sobel of generator as pt and nifti
    torch.save(
        fake_2.to("cpu"),
        os.path.join(
            tensor_path,
            f"{participant_id}_{session_id}_image-0_{vae_maps_dir.name}_output.pt",
        ),
    )
    # torch.save(
    #     residual.to("cpu"),
    #     os.path.join(
    #         tensor_path,
    #         f"{participant_id}_{session_id}_image-0_residual_{mode}.pt",
    #     ),
    # )
    # nib.save(
    #     nib.Nifti1Image(
    #         real_1_w.squeeze().to("cpu").numpy(),
    #         header=test_nii.header,
    #         affine=test_nii.affine,
    #     ),
    #     os.path.join(
    #         nifti_path,
    #         f"{participant_id}_{session_id}_hybrid_sobel_{mode}.nii.gz",
    #     ),
    # )
    nib.save(
        nib.Nifti1Image(
            fake_2.squeeze().to("cpu").numpy(),
            header=test_nii.header,
            affine=test_nii.affine,
        ),
        os.path.join(
            nifti_path,
            f"{participant_id}_{session_id}_{vae_maps_dir.name}_output.nii.gz",
        ),
    )
    # nib.save(
    #     nib.Nifti1Image(
    #         residual.squeeze().to("cpu").numpy(),
    #         header=test_nii.header,
    #         affine=test_nii.affine,
    #     ),
    #     os.path.join(
    #         nifti_path,
    #         f"{participant_id}_{session_id}_residual_{mode}.nii.gz",
    #     ),
    # )
