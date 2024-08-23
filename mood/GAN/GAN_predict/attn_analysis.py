import sys
sys.path.append('../')
from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage
from mood.GAN.GAN_trainer.utils import MinMaxNormalization, ResizeInterpolation, ResidualMean, ThresholdNormalization,index_split
import torchvision.transforms as transfroms
from pathlib import Path
from torch.utils.data import DataLoader, StackDataset, Subset
import os

CAPS_DIR=Path("/root_dir/data/brain/caps_brain_custom")
PREPROCESSING_JSON=Path("/root_dir/misc/MS_extract.json")
LABEL_TSV=Path("/root_dir/data/brain/caps_brain_t1/subjects_sessions.tsv")
SPLIT_DIR=Path("/root_dir/data/brain/caps_brain_t1/split/5_fold")

split = 0

transforms = transfroms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])

# return_dataset T1 weighting image and Pet images
t1_weighting_caps = CapsDatasetImage(
    CAPS_DIR,
    LABEL_TSV,
    PREPROCESSING_JSON,
    train_transformations= transforms
)

pet_weighting_caps = CapsDatasetImage(
    CAPS_DIR,
    LABEL_TSV,
    PREPROCESSING_JSON,
    train_transformations= transforms
)

stacked_dataset = StackDataset(t1_weighting_caps,pet_weighting_caps)


# Validation Subset
val_split_path = os.path.join(SPLIT_DIR, "split-" + str(split),"validation_baseline.tsv")
val_index = index_split(LABEL_TSV,val_split_path)
validation_dataset = Subset(stacked_dataset,val_index) 

import torch
from mood.GAN.GAN_trainer.model_cd import Generator
from mood.GAN.GAN_trainer.model import AttU_Net
from pathlib import Path
import numpy as np
GEN_PRETRAINED_DIR=Path("/root_dir/maps/GANs/pix2pix/attn_unet_25")

architecture_G = AttU_Net()
# define the Generator model
optimizer_generator = torch.optim.Adam(
    architecture_G.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler_G = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Generator_CD1 = Generator(architecture_G,
optimizer_generator,
torch.nn.L1Loss(),
scheduler_G).to(device)

Generator_CD1.load(os.path.join(GEN_PRETRAINED_DIR,"5_fold", "split-" + str(split),
                                    "best_loss", "gan", "generator", "modelCD.pth.tar"))

with torch.no_grad():
    source, target = validation_dataset[0]
    real_1 = source["image"].unsqueeze(0).to(device)
    fake_2 = Generator_CD1.forward(real_1)

torch.save(real_1.to("cpu"),"./tensor/test_input_sobel.pt")
torch.save(fake_2.to("cpu"),"./tensor/test_output_sobel.pt")

import nibabel as nib
nib.save(nib.Nifti1Image(real_1.to("cpu").numpy(), np.eye(4)), "./nifti/test_input_sobel.nii.gz")
nib.save(nib.Nifti1Image(fake_2.to("cpu").numpy(), np.eye(4)), "./nifti/test_input_sobel.nii.gz")