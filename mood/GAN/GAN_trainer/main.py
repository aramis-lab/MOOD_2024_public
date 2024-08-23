
#%%

import torch
import torchvision.transforms as transfroms
from pathlib import Path
from torch.utils.data import DataLoader, StackDataset, Subset
import json

#caps path definition

caps_directory = Path("/root_dir/datasets/caps/caps_pet_uniform")

# preprocessing_json file - 2 needed (T1 and PET)
t1_preprocessing_json = Path("/root_dir/gan/inputs/json_files/t1-linear_crop-True_mode-image.json")
pet_preprocessing_json = Path("/root_dir/gan/inputs/json_files/extract_pet_uniform_image.json")

label_path = Path("/root_dir/gan/inputs/tsv_file/labels_after_QC.tsv")

split_path = Path("/root_dir/gan/inputs/tsv_file/split/5_fold")
#%%

from clinicadl.utils.caps_dataset.caps_dataset_refactoring.caps_dataset import CapsDatasetImage
from mood.GAN.GAN_trainer.utils import MinMaxNormalization, ResizeInterpolation, ResidualMean, ThresholdNormalization

transforms_t1w = transfroms.Compose([MinMaxNormalization(),
                                      ResizeInterpolation((128,128,128)),
                                      MinMaxNormalization()])


transforms_pet = transfroms.Compose([ThresholdNormalization(),
                                    MinMaxNormalization(),
                                    ResizeInterpolation((128,128,128)),
                                    MinMaxNormalization()])

# return_dataset T1 weighting image and Pet images
t1_weighting_caps = CapsDatasetImage(
    caps_directory,
    label_path,
    t1_preprocessing_json,
    train_transformations= transforms_t1w
)


pet_weighting_caps = CapsDatasetImage(
        caps_directory,
        label_path,
        pet_preprocessing_json,
        train_transformations= transforms_pet
)

stacked_dataset = StackDataset(t1_weighting_caps,pet_weighting_caps)

#%%
#------------------
# Define parameters
#------------------
import argparse

parser = argparse.ArgumentParser(description='image synthesis from T1 to PET')

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
    choices=['conv_patch', 'patch']
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

## read command line arguments

i = args.split

# name of the output


if args.generator_pretrained != '.':
    
    generator_pretrained_epoch = int(generator_pretrained.split('_')[-1]) #generator trained have been saved in a folder: 'generator_name' + '_' + 'num_epoch'
else:
    generator_pretrained_epoch = 0

if args.discriminator_pretrained != '.':
    print(discriminator_pretrained)
    discriminator_pretrained_epoch = int(discriminator_pretrained.split('_')[-1]) #discriminator_pretrained trained have been saved in a folder: 'generator_name' + '_' + 'num_epoch'
else:
    discriminator_pretrained_epoch = 0

scheduler = args.scheduler

#%%

#########  Training and Validation dataset  ########
from mood.GAN.GAN_trainer.utils import index_split

split_i_path = os.path.join(split_path,"split-" + str(i))
# Training Subset
train_split_path = os.path.join(split_i_path,"train.tsv")
train_index = index_split(label_path, train_split_path)
training_dataset = Subset(stacked_dataset,train_index)

# Validation Subset
val_split_path = os.path.join(split_i_path,"validation_baseline.tsv")
val_index = index_split(label_path,val_split_path)
validation_dataset = Subset(stacked_dataset,val_index) 

#% Dataloader

import os
from mood.GAN.GAN_trainer.model import Unet, Discriminator64, AttU_Net, NLayerDiscriminator
from mood.GAN.GAN_trainer.model_cd import Generator, ConvPatchD, DecomposedD

if useDDP:
    import idr_torch 
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=idr_torch.size, 
                        rank=idr_torch.rank)

    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")

    if args.generator_name == 'Unet':
        architecture_G_t = Unet().to(gpu)
    elif args.generator_name == 'AttU_Net':
        architecture_G_t = AttU_Net().to(gpu)

    if args.discriminator_name == "patch":
        architecture_D_t = Discriminator64().to(gpu)
    elif args.discriminator_name == "conv_patch":
        architecture_D_t = NLayerDiscriminator(input_nc=2).to(gpu)

    architecture_G = DDP(architecture_G_t,device_ids=[idr_torch.local_rank])

    arch
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset,
                                                              num_replicas=idr_torch.size,
                                                              rank=idr_torch.rank,
                                                              shuffle=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,
                                                              num_replicas=idr_torch.size,
                                                              rank=idr_torch.rank,
                                                              shuffle=False)



    val_loader = DataLoader(
            validation_dataset,
            batch_size= batch_size_per_gpu,
            shuffle = False,
            num_workers = args.n_proc,
            pin_memory= True,
            prefetch_factor= 2
        )

else:
    train_loader = DataLoader(
        training_dataset,
        batch_size= args.batch_size,
        shuffle = True,
        num_workers = args.n_proc,
        pin_memory= True,
        )

    val_loader = DataLoader(
        validation_dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.n_proc,
        pin_memory= True,
        prefetch_factor= 2
        )

    if args.generator_name == 'Unet':
            architecture_G = Unet().to(device)
    elif args.generator_name == 'AttU_Net':
        architecture_G = AttU_Net().to(device)

    if args.discriminator_name == "patch":
        architecture_D = Discriminator64()
    elif args.discriminator_name == "conv_patch":
        architecture_D = NLayerDiscriminator(input_nc=2)


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
    
    output_results_fold = os.path.join(args.output_results, name_output, "5_fold","split-" + str(i))

    trainer_option = TrainerOption(
        n_epoch = args.n_epoch,
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
        Discriminator_CD= Discriminator_CD1,
        opts_training= trainer_option,
    )

    Trainer.train(TrainLoader= train_loader,
                ValLoader= val_loader )

elif model == 'conditional_gan':

        # define the Generator model
    optimizer_generator = torch.optim.Adam(
        architecture_G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_G = None
    
    Generator_CD1 = Generator(architecture_G,
    optimizer_generator,
    torch.nn.L1Loss(),
    scheduler_G)

    Generator_CD1.load(args.generator_pretrained)


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
    Discriminator_CD.load(args.discriminator_pretrained)

    if args.name_output is not None:
        name_output = args.name_output
    else:
        name_output = args.generator_name + '_' + str(args.n_epoch)
    
    output_results_fold = os.path.join(args.output_results, name_output, "5_fold","split-" + str(i))

    trainer_option = TrainerOption(
        n_epoch = args.n_epoch,
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
    criterion_GAN = args.criterion_GAN,
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

elif model == ['resume_training_gan']:

    if args.name_output is not None:
        name_output = args.name_output
    else:
         raise "Error in the name"
    
    output_results_fold = os.path.join(output_results, name_output, "5_fold","split-" + str(i))
    f = open(os.path.join(output_results_fold,"parameters.json"))
    opts_train = json.load(f)
    model_generator_name = opts_train["generator"]
    model_discriminator_name = opts_train["discriminator"]

    if model_generator_name == 'Unet':
            model_generator = Unet()
    elif model_generator_name == 'AttU_Net':
        model_generator = AttU_Net()

    if model_discriminator_name == "patch":
        model_discriminator = Discriminator64()
    elif model_discriminator_name == "conv_patch":
        model_discriminator = NLayerDiscriminator(input_nc=2)
    print("let's gooooo")
    generator = resume_training_gan(train_loader, val_loader, model_generator, model_discriminator, output_results_fold, opts_train, caps_directory)
