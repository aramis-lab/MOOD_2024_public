
#%%

import torch
import torchvision.transforms as transfroms
from pathlib import Path
from torch.utils.data import DataLoader, StackDataset, Subset
import json

#%%
#------------------
# Define parameters
#------------------
import argparse

parser = argparse.ArgumentParser(description='image synthesis from T1 to PET')

parser.add_argument(
        'caps_directory',
        help='Where results will be saved',
        default=None
    )

parser.add_argument(
        'preprocessing_json',
        help='Where results will be saved',
        default=None
    )

parser.add_argument(
        'label_tsv', 
        help='Where results will be saved',
        default=None
    )

parser.add_argument(
        'split_dir',
        help='Where results will be saved',
        default=None
    )

parser.add_argument(
        'output_results',
        help='Where results will be saved',
        default=None
    )

parser.add_argument(
    'training',
    help='Name of the type of the model used',
    default='generator', type=str,
    choices=['generator', 'conditional_gan', 'discriminator','resume_training_gan']
)

parser.add_argument(
    '--split',
    type=int,
    help='split number',
    default=0 
)

parser.add_argument(
    '--name_output',
    type=str,
    default=None
)

parser.add_argument(
    '--generator_name',
    help='Name of the type of the model used',
    default='Unet', type=str,
    choices=['Unet', 'AttU_Net']
)

parser.add_argument(
    '--discriminator_name',
    help='Name of the type of the model used',
    default='conv_patch', type=str,
    choices=['conv_patch', 'decomposed']
)


parser.add_argument(
    '--generator_pretrained',
    help='Path to the pretrained generator',
    default='.', type=str,
)

parser.add_argument(
    '--discriminator_pretrained',
    help='Path to the pretrained discriminator',
    default='.', type=str,
)

parser.add_argument(
        '--n_epoch',
        type=int,
        default=300,
        help='number of epoch'
    )

parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning_rate'
    )

parser.add_argument(
    '--beta1',
    type=float,
    default=0.5, 
    help='beta1 for Adam Optimizer'
)

parser.add_argument(
    '--beta2',
    type=float,
    default=0.999, 
    help='beta1 for Adam Optimizer'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=1, 
    help='batch_size'
)

parser.add_argument(
    '--n_gpu',
    type=int,
    default=1, 
    help='number_id_gpu'
)

parser.add_argument(
    '--n_proc',
    type=int,
    default=3, 
    help='number_id_cpu'
)

parser.add_argument(
    '--scheduler',
    type=str, 
    default='none',
    help='Scheduler'
)

parser.add_argument(
    '--lambda_GAN', 
    type=float,
    default= 1.,
    help= 'Weights criterion_GAN in the generator loss'
)

parser.add_argument(
    '--lambda_pixel', 
    type=float,
    default= 1.,
    help= 'Weights criterion_pixelwise in the generator loss'
)

parser.add_argument(
    '--criterion_gan', 
    type=str,
    default='MSE',
    choices=['BCE','MSE'],
    help= 'Adversial Loss definition BinaryCrossEntropy or MSE GAN'
)

parser.add_argument(
    '--criterion_pixel',
    type=str,
    default='L1',
    choices=['L1','SSIM'],
    help= 'Pixelwise Loss definition'
)

parser.add_argument(
    '--mask_brain',
    type=bool,
    default=False,
    help= 'mask_brain'
)

args = parser.parse_args()

caps_directory = args.caps_directory
label_path = args.label_tsv
split_i_path = args.split_dir


#%%

from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage
from mood.GAN.GAN_trainer.utils import MinMaxNormalization, ResizeInterpolation, ResidualMean, ThresholdNormalization

transforms = transfroms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])





from skimage.filters import sobel
class Sobel(object):

    def __call__(self,image):
        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0)
import numpy as np
import skimage.morphology as MM
from skimage.filters import gaussian
class AugmentationSobel(object):

    def __call__(self,image):

        isaugmented = np.random.randint(0,2, dtype=bool)

        if isaugmented:
            import skimage.morphology as MM

            radius = np.random.randint(20,60)
            mask_bool = np.zeros_like(image.squeeze().numpy())
            mask = MM.ball(radius)

            anom_pos = np.random.randint(0 + radius + 3 ,128 - radius -1 ,size = (1,3))
            anom_pos = anom_pos[0]

            mask_bool[anom_pos[0]-radius:anom_pos[0]+radius+1,
                                        anom_pos[1]-radius:anom_pos[1]+radius+1,
                                        anom_pos[2]-radius:anom_pos[2]+radius+1] = mask


            ga_image = gaussian(image, sigma = 4)

            image.squeeze()[mask_bool == 1] = torch.from_numpy(ga_image.squeeze())[mask_bool == 1]

            return image.unsqueeze(0)
        else:
            return image


transforms_input = transfroms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization(),
                                      AugmentationSobel(),
                                      Sobel()])

# return_dataset T1 weighting image and Pet images
#t1_weighting_caps = CapsDatasetImage(
#    Path(args.caps_directory),
#    Path(args.label_tsv),
#    Path(args.preprocessing_json),
#    train_transformations= transforms_input
#)

t1_weighting_caps = CapsDatasetImage(
    Path("/root_dir/data/brain/caps_brain_custom"),
    Path(args.label_tsv),
    Path("/root_dir/misc/MS_extract.json"),
    train_transformations= transforms_input
)

pet_weighting_caps = CapsDatasetImage(
    Path("/root_dir/data/brain/caps_brain_custom"),
    Path(args.label_tsv),
    Path("/root_dir/misc/MS_extract.json"),
    train_transformations= transforms
)

stacked_dataset = StackDataset(t1_weighting_caps,pet_weighting_caps)


## read command line arguments

i = args.split

# name of the output


if args.generator_pretrained != '.':
    
    generator_pretrained_epoch = int(args.generator_pretrained.split('_')[-1]) #generator trained have been saved in a folder: 'generator_name' + '_' + 'num_epoch'
else:
    generator_pretrained_epoch = 0

if args.discriminator_pretrained != '.':
    discriminator_pretrained_epoch = int(args.discriminator_pretrained.split('_')[-1]) #discriminator_pretrained trained have been saved in a folder: 'generator_name' + '_' + 'num_epoch'
else:
    discriminator_pretrained_epoch = 0

scheduler = args.scheduler

#%%
import os
from mood.GAN.GAN_trainer.model import Unet, Discriminator64, AttU_Net, NLayerDiscriminator
from mood.GAN.GAN_trainer.model_cd import Generator, ConvPatchD, DecomposedD

if args.generator_name == 'Unet':
        architecture_G = Unet()
elif args.generator_name == 'AttU_Net':
    architecture_G = AttU_Net()

if args.discriminator_name == "decomposed":
    architecture_D = Discriminator64()
elif args.discriminator_name == "conv_patch":
    architecture_D = NLayerDiscriminator(input_nc=2)

#########  Training and Validation dataset  ########
from mood.GAN.GAN_trainer.utils import index_split

split_i_path = os.path.join(args.split_dir,"split-" + str(i))
# Training Subset
label_path = args.label_tsv

train_split_path = os.path.join(split_i_path,"train.tsv")
train_index = index_split(label_path, train_split_path)
training_dataset = Subset(stacked_dataset,train_index)

# Validation Subset
val_split_path = os.path.join(split_i_path,"validation_baseline.tsv")
val_index = index_split(label_path,val_split_path)
validation_dataset = Subset(stacked_dataset,val_index) 

#% Dataloader

train_loader = DataLoader(
    training_dataset,
    batch_size= args.batch_size,
    shuffle = True,
    num_workers = args.n_proc,
    pin_memory= True
)

val_loader = DataLoader(
    validation_dataset,
    batch_size= args.batch_size,
    shuffle = True,
    num_workers = args.n_proc,
    pin_memory= True
)


from mood.GAN.GAN_trainer.trainer import GAN_trainer
from mood.GAN.GAN_trainer.config_trainer import TrainerOption, JobOption

if args.training == 'generator':

    # define the Generator model
    optimizer_generator = torch.optim.Adam(
        architecture_G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_G = None
    
    Generator_CD1 = Generator(architecture_G,
    optimizer_generator,
    torch.nn.L1Loss(),
    scheduler_G)

    if args.name_output is not None:
        name_output = args.name_output
    else:
        name_output = args.generator_name + '_' + str(args.n_epoch)
    
    output_results_fold = os.path.join(args.output_results, name_output, "split-" + str(i))

    trainer_option = TrainerOption(
        n_epoch = args.n_epoch,
        current_epoch = 0,
        TrainGen = True,
        TrainDisc = False,
        output_results = Path(output_results_fold)
    )

    option_job = JobOption(
    task = args.training,
    caps_dataset = caps_directory,
    label_path = label_path,
    split_path = split_i_path,
    output_folder = output_results_fold,
    generator_name = args.generator_name,
    discriminator_name = None,
    generator_pretrained_epoch = generator_pretrained_epoch,
    discriminator_pretrained_epoch = discriminator_pretrained_epoch,
    #optimizer
    optimizer_lr = args.lr,
    optimizer_b1 = args.beta1,
    optimizer_b2 = args.beta2,

    scheduler = args.scheduler,
    lambda_GAN = None,
    lambda_pixel = None,
    criterion_GAN = None,
    criterion_pixelwise = args.criterion_pixel,
    # epochs
    n_epoch = args.n_epoch,
    split =  i,
    batch_size = args.batch_size,

    mask_brain = args.mask_brain,
    # 
    n_proc = args.n_proc,
    n_gpu = args.n_gpu
    )
    
    
    if not os.path.exists(output_results_fold):
            os.makedirs(output_results_fold)

    with open(os.path.join(output_results_fold,'parameters.json'), 'w') as fp:
        fp.write(option_job.model_dump_json(indent = 2))
        #option_job.json()
        #json.dump(option_job.model_dump(),fp, indent=2)

    Trainer = GAN_trainer(
        Generator_CD= Generator_CD1,
        Discriminator_CD= None,
        opts_training= trainer_option,
    )

    Trainer.train(TrainLoader= train_loader,
                ValLoader= val_loader )

elif args.training == 'conditional_gan':


        # define the Generator model
    optimizer_generator = torch.optim.Adam(
        architecture_G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_G = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Generator_CD1 = Generator(architecture_G,
    optimizer_generator,
    torch.nn.L1Loss(),
    scheduler_G).to(device)

    Generator_CD1.load(os.path.join(args.generator_pretrained, "split-" + str(i),
                                     "best-loss", "gan", "generator", "modelCD.pth.tar"))


    optimizer_D = torch.optim.Adam(
        architecture_D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_D = None
    
    if args.discriminator_name == "decomposed":
        Discriminator_CD = DecomposedD(architecture_D,
                                        optimizer_D,
                                        torch.nn.MSELoss(),
                                        scheduler_D
                                        ).to(device)
    elif args.discriminator_name == "conv_patch":
        Discriminator_CD = PatchConv(architecture_D,
                                        optimizer_D,
                                        torch.nn.MSELoss(),
                                        scheduler_D
                                        ).to(device)
    Discriminator_CD.load(os.path.join(args.discriminator_pretrained, "split-" + str(i),
                                     "best-loss", "gan", "discriminator", "modelCD.pth.tar"))
    if args.name_output is not None:
        name_output = args.name_output
    else:
        name_output = args.generator_name + '_' + str(args.n_epoch)
    
    output_results_fold = os.path.join(args.output_results, name_output, "split-" + str(i))

    trainer_option = TrainerOption(
        n_epoch = args.n_epoch,
        current_epoch = 0,
        TrainGen = True,
        TrainDisc = True,
        output_results = Path(output_results_fold)
    )

    option_job = JobOption(
    task = args.training,
    caps_dataset = caps_directory,
    label_path = label_path,
    split_path = split_i_path,
    output_folder = output_results_fold,
    generator_name = args.generator_name,
    discriminator_name = args.discriminator_name,
    generator_pretrained_epoch = generator_pretrained_epoch,
    discriminator_pretrained_epoch = discriminator_pretrained_epoch,
    #optimizer
    optimizer_lr = args.lr,
    optimizer_b1 = args.beta1,
    optimizer_b2 = args.beta2,

    scheduler = args.scheduler,
    lambda_GAN = args.lambda_GAN,
    lambda_pixel = args.lambda_pixel,
    criterion_GAN = args.criterion_gan,
    criterion_pixelwise = args.criterion_pixel,
    # epochs
    n_epoch = args.n_epoch,
    split =  i,
    batch_size = args.batch_size,

    mask_brain = args.mask_brain,
    # 
    n_proc = args.n_proc,
    n_gpu = args.n_gpu
    )
    
    
    if not os.path.exists(output_results_fold):
            os.makedirs(output_results_fold)

    with open(os.path.join(output_results_fold,'parameters.json'), 'w') as fp:
        fp.write(option_job.model_dump_json(indent = 2))
        #option_job.json()
        #json.dump(option_job.model_dump(),fp, indent=2)

    Trainer = GAN_trainer(
        Generator_CD= Generator_CD1,
        Discriminator_CD= Discriminator_CD,
        opts_training= trainer_option,
    )

    Trainer.train(TrainLoader= train_loader,
                ValLoader= val_loader )


elif args.training == 'discriminator':

        # define the Generator model
    optimizer_generator = torch.optim.Adam(
        architecture_G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_G = None
    
    Generator_CD1 = Generator(architecture_G,
    optimizer_generator,
    torch.nn.L1Loss(),
    scheduler_G)

    Generator_CD1.load(os.path.join(args.generator_pretrained, "split-" + str(i),
                                     "best-loss", "gan", "generator", "modelCD.pth.tar"))


    optimizer_D = torch.optim.Adam(
        architecture_D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_D = None
    
    if args.discriminator_name == "decomposed":
        Discriminator_CD = DecomposedD(architecture_D,
                                        optimizer_D,
                                        torch.nn.MSELoss(),
                                        scheduler_D
                                        )
    elif args.discriminator_name == "conv_patch":
        Discriminator_CD = PatchConv(architecture_D,
                                        optimizer_D,
                                        torch.nn.MSELoss(),
                                        scheduler_D
                                        )

    if args.name_output is not None:
        name_output = args.name_output
    else:
        name_output = args.generator_name + '_' + str(args.n_epoch)
    
    output_results_fold = os.path.join(args.output_results, name_output, "split-" + str(i))

    trainer_option = TrainerOption(
        n_epoch = args.n_epoch,
        current_epoch = 0,
        TrainGen = False,
        TrainDisc = True,
        output_results = Path(output_results_fold)
    )

    option_job = JobOption(
    task = args.training,
    caps_dataset = caps_directory,
    label_path = label_path,
    split_path = split_i_path,
    output_folder = output_results_fold,
    generator_name = args.generator_name,
    discriminator_name = args.discriminator_name,
    generator_pretrained_epoch = generator_pretrained_epoch,
    discriminator_pretrained_epoch = discriminator_pretrained_epoch,
    #optimizer
    optimizer_lr = args.lr,
    optimizer_b1 = args.beta1,
    optimizer_b2 = args.beta2,

    scheduler = args.scheduler,
    lambda_GAN = args.lambda_GAN,
    lambda_pixel = args.lambda_pixel,
    criterion_GAN = args.criterion_gan,
    criterion_pixelwise = args.criterion_pixel,
    # epochs
    n_epoch = args.n_epoch,
    split =  i,
    batch_size = args.batch_size,

    mask_brain = args.mask_brain,
    # 
    n_proc = args.n_proc,
    n_gpu = args.n_gpu
    )
    
    
    if not os.path.exists(output_results_fold):
            os.makedirs(output_results_fold)

    with open(os.path.join(output_results_fold,'parameters.json'), 'w') as fp:
        fp.write(option_job.model_dump_json(indent = 2))
        #option_job.json()
        #json.dump(option_job.model_dump(),fp, indent=2)

    Trainer = GAN_trainer(
        Generator_CD= Generator_CD1,
        Discriminator_CD= Discriminator_CD,
        opts_training= trainer_option,
    )

    Trainer.train(TrainLoader= train_loader,
                ValLoader= val_loader )

elif args.training == ['resume_training_gan']:

    if args.name_output is not None:
        name_output = args.name_output
    else:
        "no model name specified"
    
    output_results_fold = os.path.join(args.output_results, name_output, "split-" + str(i))


    f = open(os.path.join(output_results_fold,"parameters.json"))
    opts_train = json.load(f)
    model_generator_name = opts_train["generator_name"]

   

    if model_generator_name == 'Unet':
            architecture_G = Unet()
    elif model_generator_name == 'AttU_Net':
        architecture_G = AttU_Net()

    
       # define the Generator model
    optimizer_generator = torch.optim.Adam(
        architecture_G.parameters(), lr=opts_train["optimizer_lr"], betas=(opts_train["optimizer_b1"], opts_train["optimizer_b2"]))
    scheduler_G = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Generator_CD1 = Generator(architecture_G,
    optimizer_generator,
    torch.nn.L1Loss(),
    scheduler_G).to(device)

    Generator_CD1.load(os.path.join(output_results_fold,
                                     "checkpoint", "gan", "generator", "modelCD.pth.tar"))

    if opts_train["task"] != "generator":
        model_discriminator_name = opts_train["discriminator_name"]

        if model_discriminator_name == "decomposed":
            architecture_D = Discriminator64()
        elif model_discriminator_name == "conv_patch":
            architecture_D = NLayerDiscriminator(input_nc=2)

        optimizer_D = torch.optim.Adam(
            architecture_D.parameters(), lr=opts_train["optimizer_lr"], betas=(opts_train["optimizer_b1"], opts_train["optimizer_b2"]))
        scheduler_D = None
    
        if model_discriminator_name == "decomposed":
            Discriminator_CD = DecomposedD(architecture_D,
                                            optimizer_D,
                                            torch.nn.MSELoss(),
                                            scheduler_D
                                            ).to(device)
        elif model_discriminator_name == "conv_patch":
            Discriminator_CD = PatchConv(architecture_D,
                                            optimizer_D,
                                            torch.nn.MSELoss(),
                                            scheduler_D
                                            ).to(device)

        Discriminator_CD.load(os.path.join(output_results_fold,
                                        "checkpoint", "gan", "discriminator", "modelCD.pth.tar"))
    else:
        Discriminator_CD = None
    


    f = open(os.path.join(output_results_fold,"checkpoint","training_params.json"))
    epoch_info = json.load(f)
    trainer_option = TrainerOption(
        n_epoch = opts_train["n_epoch"],
        current_epoch = epoch_info["current_epoch"],
        TrainGen = True,
        TrainDisc = True,
        output_results = Path(output_results_fold)
    )
    

    Trainer = GAN_trainer(
        Generator_CD= Generator_CD1,
        Discriminator_CD= Discriminator_CD,
        opts_training= trainer_option,
    )

    Trainer.train(TrainLoader= train_loader,
                ValLoader= val_loader )