#%%
import torch
import sys
sys.path.append('../')
from mood.GAN.GAN_trainer.model_cd import Generator
from mood.GAN.GAN_trainer.model import AttU_Net
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description='image synthesis from T1 to PET')

parser.add_argument(
    '--split',
    type=int,
    help='split number',
    default=0 
)
args = parser.parse_args()
split = args.split

CAPS_DIR=Path("/root_dir/data/brain/caps_test")
LABEL_TSV=Path("/root_dir/data/brain/caps_test/subjects_sessions.tsv")
GEN_PRETRAINED_DIR=Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_25")
PREPROCESSING_JSON=Path("/root_dir/misc/MS_extract.json")


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

split_i_path = os.path.join(GEN_PRETRAINED_DIR,"5_fold", "split-" + str(split), "best_loss")

print(split)
print(split_i_path)
from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage
from GAN_trainer.utils import MinMaxNormalization, ResizeInterpolation, ResidualMean, ThresholdNormalization

from skimage.filters import sobel
class Sobel(object):

    def __call__(self,image):
        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0)


import torchvision.transforms as transforms
transforms_input = transforms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization(), Sobel()])

transforms_output = transforms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])


# return_dataset T1 weighting image and Pet images
t1_weighting_caps_input = CapsDatasetImage(
    Path(CAPS_DIR),
    Path(LABEL_TSV),
    Path(PREPROCESSING_JSON),
    train_transformations= transforms_input
)

t1_weighting_caps_output = CapsDatasetImage(
    Path(CAPS_DIR),
    Path(LABEL_TSV),
    Path(PREPROCESSING_JSON),
    train_transformations= transforms_output
)
from torch.utils.data import DataLoader, StackDataset
stacked_dataset = StackDataset(t1_weighting_caps_input,t1_weighting_caps_output)

val_loader = DataLoader(
            stacked_dataset,
            batch_size= 1,
            shuffle = False,
            num_workers = 10,
            pin_memory= True,
            prefetch_factor= 2)

import nibabel as nib

for i, (source,target) in enumerate(val_loader):
    print(f"iteration:  {i} / {len(val_loader)} ")

    participant_id = source["participant_id"][0]
    session_id = source["session_id"][0]


    real_1 = source["image"].to(device)
    real_2 = target["image"].to(device)

    session_path = os.path.join(split_i_path, "CapsOutput",
    "subjects", participant_id, session_id)

    nifti_path = os.path.join(session_path,"custom")
    os.makedirs(nifti_path)
    tensor_path = os.path.join(session_path,"deeplearning_prepare_data","image_based","custom")
    os.makedirs(tensor_path)
    with torch.no_grad():
        fake_2 = Generator_CD1.forward(real_1)
        residual = real_2 - fake_2
    
    # save torch_tensor
    torch.save(real_1.to("cpu"), os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_sobel.pt'))
    torch.save(real_2.to("cpu"), os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_input.pt'))
    torch.save(fake_2.to("cpu"), os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_output.pt'))
    torch.save(residual.to("cpu"), os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_residual.pt'))

    #save niifti
    niftii_input = os.path.join(CAPS_DIR, "subjects", participant_id, session_id, "custom",
    participant_id + "_" + session_id + "_mood.nii.gz")
    test_nii = nib.load(niftii_input)

    nib.save(nib.Nifti1Image(real_1.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_sobel.nii.gz'))
    nib.save(nib.Nifti1Image(real_2.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_input.nii.gz'))
    nib.save(nib.Nifti1Image(fake_2.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_output.nii.gz'))
    nib.save(nib.Nifti1Image(residual.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_residual.nii.gz'))


          