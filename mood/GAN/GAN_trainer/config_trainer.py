from pydantic import BaseModel, model_serializer
from typing import Union
from pathlib import Path

class TrainerOption(BaseModel):
    n_epoch : int
    current_epoch : int
    TrainGen : bool
    TrainDisc : Union[bool, None]
    output_results : Path

class JobOption(BaseModel):

    task : str
    caps_dataset : Union[str, Path]
    label_path : Union[str,Path]
    split_path : Union[str,Path]
    output_folder : Union[str, Path]

    generator_name : Union[str, None]
    discriminator_name : Union[str, None]

    generator_pretrained_epoch : Union[int, None]
    discriminator_pretrained_epoch : Union[int, None]

    #optimizer
    optimizer_lr : float
    optimizer_b1 : float
    optimizer_b2 : float

    scheduler : Union[str, None]
    lambda_GAN : Union[float, None]
    lambda_pixel : Union[float, None]
    criterion_GAN : Union[str, None]
    criterion_pixelwise : Union[str, None]
    # epochs
    n_epoch : int
    split : int
    batch_size : int

    mask_brain : bool
    # 
    n_proc : int
    n_gpu : int
