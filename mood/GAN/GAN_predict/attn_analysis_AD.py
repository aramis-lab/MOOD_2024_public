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

from skimage.filters import sobel
class Sobel(object):

    def __call__(self,image):
        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0)



transforms_input = transfroms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization(), Sobel()])

transforms_output = transfroms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])


import torch
from mood.GAN.GAN_trainer.model_cd import Generator
from mood.GAN.GAN_trainer.model import AttU_Net
from pathlib import Path
import numpy as np
GEN_PRETRAINED_DIR=Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_25")

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
    real_1 = transforms_input(torch.load("./tensor/sub-224_ses-M003_mood.pt")).unsqueeze(0).to(device)
    real_2 = transforms_output(torch.load("./tensor/sub-224_ses-M003_mood.pt")).unsqueeze(0).to(device)
    fake_2 = Generator_CD1.forward(real_1)

torch.save(real_1.to("cpu"),"./tensor/sub-224_ses-M003_mood_input_sobel.pt")
torch.save(fake_2.to("cpu"),"./tensor/sub-224_ses-M003_mood_output_sobel.pt")
torch.save((real_2 - fake_2).to("cpu"),"./tensor/sub-224_ses-M003_mood_res_sobel.pt")


test_path = Path("/root_dir/maps/MAPS_MS_BetaVAE_0/split-4/best-loss/CapsOutput/subjects/sub-037/ses-M000/custom/sub-037_ses-M000_residual_-mean_validation.nii.gz")
import nibabel as nib
test_nii = nib.load(test_path)

nib.save(nib.Nifti1Image((real_2 - fake_2).squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine), "./nifti/224_ses_test_res_sobel.nii.gz")
nib.save(nib.Nifti1Image(real_1.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine), "./nifti/224_ses_test_input_sobel.nii.gz")
nib.save(nib.Nifti1Image(real_2.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine), "./nifti/224_ses_test_GT_sobel.nii.gz")
nib.save(nib.Nifti1Image(fake_2.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine), "./nifti/224_ses_test_output_sobel.nii.gz")