#%%
import torch
import sys
sys.path.append('../')
from mood.GAN.GAN_trainer.model_cd import Generator
from mood.GAN.GAN_trainer.model import AttU_Net
from pathlib import Path
import os
import skimage.morphology as MM
import numpy as np
import nibabel as nib
from copy import deepcopy
from skimage.metrics import structural_similarity
from skimage.filters import gaussian
from skimage.exposure import match_histograms

split = 4

CAPS_DIR=Path("/root_dir/maps/MAPS_MS_BetaVAE_0/split-4/best-loss/CapsOutput")
LABEL_TSV=Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100/groups/test/data.tsv")
PREPROCESSING_JSON=Path("/root_dir/misc/VAE_maps_extract.json")
GEN_PRETRAINED_DIR=Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100")


architecture_G = AttU_Net()
# define the Generator model
optimizer_generator = torch.optim.Adam(
    architecture_G.parameters(), lr=0.0002, betas=(0.9, 0.999))
scheduler_G = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
Generator_CD1 = Generator(architecture_G,
optimizer_generator,
torch.nn.L1Loss(),
scheduler_G).to(device)

Generator_CD1.load(os.path.join(GEN_PRETRAINED_DIR, "split-" + str(split),
                                    "best-loss", "gan", "generator", "modelCD.pth.tar"))

split_i_path = os.path.join(GEN_PRETRAINED_DIR, "split-" + str(split), "best-loss")

print(split)
print(split_i_path)
from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage
from mood.GAN.GAN_trainer.utils import MinMaxNormalization, ResizeInterpolation, ResidualMean, ThresholdNormalization


from skimage.filters import sobel
class Sobel(object):

    def __call__(self,image):
        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0)


import torchvision.transforms as transforms
transforms_input = transforms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])

transforms_output = transforms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])


# return_dataset T1 weighting image and Pet images
t1_weighting_caps_VAE = CapsDatasetImage(
    Path(CAPS_DIR),
    Path(LABEL_TSV),
    Path(PREPROCESSING_JSON),
    train_transformations= transforms_output
)

t1_weighting_caps_input = CapsDatasetImage(
    Path("/root_dir/data/brain/caps_test"),
    #Path("/root_dir/data/brain/caps_brain_custom"),
    #Path("/root_dir/data/brain/caps_test"),
    Path(LABEL_TSV),
    Path("/root_dir/misc/MS_extract.json"),
    train_transformations= transforms_input
)

t1_weighting_caps_output = CapsDatasetImage(
    Path("/root_dir/data/brain/caps_test"),
    #Path("/root_dir/data/brain/caps_brain_custom"),
    #Path("/root_dir/data/brain/caps_test"),
    Path(LABEL_TSV),
    Path("/root_dir/misc/MS_extract.json"),
    train_transformations= transforms_output
)
from torch.utils.data import DataLoader, StackDataset
stacked_dataset = StackDataset(t1_weighting_caps_input,t1_weighting_caps_output,t1_weighting_caps_VAE)

val_loader = DataLoader(
            stacked_dataset,
            batch_size= 1,
            shuffle = False,
            num_workers = 10,
            pin_memory= True,
            prefetch_factor= 2)


def mean_correction(image_1, image_2):
    sigma_c = 2.5
    image_1  = gaussian(image_1, sigma=sigma_c).astype(np.float32)
    image_2 = mean_contour.astype(np.float32)
    image_21 = match_histograms(image_2, image_1, channel_axis=1)
    score_MEAN, weight =  structural_similarity(image_1, image_21, data_range=1, win_size=15, gradient=False, full=True, gaussian_weights=False)
    return weight

def vae_correction(image_1, image_2):
    sigma_c = 1.5
    image_1  = gaussian(image_1, sigma=sigma_c).astype(np.float32)
    image_2 = image_2.astype(np.float32)
    score_VAE, weight =  structural_similarity(image_1, image_2, data_range=1, win_size=11, gradient=False, full=True, gaussian_weights=False)
    return score_VAE, weight

input_path = Path("/root_dir/residual/input_mean.nii.gz")
mean_contour = nib.load(input_path).get_fdata()
mean_tensor =  ResizeInterpolation((128))(torch.tensor(mean_contour)).double()
mean_contour = mean_tensor.squeeze().numpy()

mask = MM.dilation(mean_contour>0.01)

for i, (source,target, vae_cop) in enumerate(val_loader):
    print(f"iteration:  {i} / {len(val_loader)} ")

    participant_id = source["participant_id"][0]
    session_id = source["session_id"][0]

    real_1 = source["image"]
    real_2 = target["image"]
    vae_image = vae_cop["image"]

    session_path = os.path.join(split_i_path, "CapsOutput",
    "subjects", participant_id, session_id)

    nifti_path = os.path.join(session_path,"custom")
    os.makedirs(nifti_path, exist_ok=True)
    tensor_path = os.path.join(session_path,"deeplearning_prepare_data","image_based","custom")
    os.makedirs(tensor_path, exist_ok=True)

     #save niifti
    niftii_input = os.path.join(CAPS_DIR, "subjects", participant_id, session_id, "custom",
    participant_id + "_" + session_id + "_input.nii.gz")
    test_nii = nib.load(niftii_input)


    torch.save(real_1, os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_input.pt'))
    nib.save(nib.Nifti1Image(real_1.squeeze().numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_input.nii.gz'))
    
    #score_SSIM, _, weight = structural_similarity(real_2.squeeze().numpy().astype(np.float32), vae_image.squeeze().numpy().astype(np.float32), data_range=1., win_size=21, gradient=True, full=True)
    #if score_SSIM > 0.5:
    #    print(score_SSIM)
    #    real_1[torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.5] = vae_image[ torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.5]
    #    torch.save(real_1, os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_hybrid.pt'))
    #    nib.save(nib.Nifti1Image(real_1.squeeze().numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_hybrid.nii.gz'))
    #
    #    real_1 = Sobel()(real_1).unsqueeze(0)
    #else:
    #    torch.save(real_1_w, os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_hybrid.pt'))
    #    nib.save(nib.Nifti1Image(real_1.squeeze().numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_hybrid.nii.gz'))
    #
    #    real_1 = Sobel()(real_1).unsqueeze(0)
    
    #checking the VAE reconstruction
    check = structural_similarity(mean_contour.astype(np.float32), vae_image.squeeze().numpy().astype(np.float32), data_range=1., win_size=11, gradient=False, full=False)
    if check < 0.7:
        weight = mean_correction(real_2.squeeze().numpy().astype(np.float32),mean_contour)
        real_1[torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.6] = (mean_tensor.unsqueeze(0)[torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.6]).float()
    else:
        score_VAE, weight = vae_correction(real_2.squeeze().numpy().astype(np.float32), vae_image.squeeze().numpy().astype(np.float32))
        if score_VAE < 0.6:
            weight = mean_correction(real_2.squeeze().numpy().astype(np.float32),mean_contour)
            real_1[torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.6] = (mean_tensor.unsqueeze(0)[ torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.6]).float()
        else:
            real_1[torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.6] = vae_image[ torch.from_numpy(weight).unsqueeze(0).unsqueeze(0) < 0.6]

    torch.save(real_1, os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_hybrid_1.pt'))
    nib.save(nib.Nifti1Image(real_1.squeeze().numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_hybrid_1.nii.gz'))
    real_1 = Sobel()(real_1).unsqueeze(0)
    
    
    real_1_w = real_1.to(device)
    
    print(real_1_w.device)
    #print(weight.shape)
    #with torch.no_grad():
        #real_1_w  = real_1 * torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).to(device)
    #print(real_1_w.dtype)
    
    #real_1_w.to(device)
    #print(torch.any(torch.isnan(real_1_w)))
    real_2 = real_2.to(device)
    
    with torch.no_grad():
        fake_2 = Generator_CD1.forward(real_1_w)
        residual = real_2 - fake_2
    
    # save torch_tensor
    
    torch.save(fake_2.to("cpu"), os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_output_1.pt'))
    torch.save(residual.to("cpu"), os.path.join(tensor_path, participant_id + '_' + session_id + '_image-0_residual_1.pt'))

   

    nib.save(nib.Nifti1Image(real_1_w.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_hybrid_sobel_1.nii.gz'))
    nib.save(nib.Nifti1Image(fake_2.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_output_1.nii.gz'))
    nib.save(nib.Nifti1Image(residual.squeeze().to("cpu").numpy(), header=test_nii.header, affine=test_nii.affine),  os.path.join(nifti_path, participant_id + '_' + session_id + '_residual_1.nii.gz'))


          