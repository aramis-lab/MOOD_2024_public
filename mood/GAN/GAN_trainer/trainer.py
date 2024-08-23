import torch
import torch.nn as nn
from mood.GAN.GAN_trainer.model_cd import ModelEvolve
from mood.GAN.GAN_trainer.config_trainer import TrainerOption
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import time 
import datetime
import os
import sys
import nibabel as nib
import numpy as np


class GAN_trainer(nn.Module):

    def __init__(self, Generator_CD : ModelEvolve , Discriminator_CD: ModelEvolve, opts_training : TrainerOption):
        
        super().__init__()
        self.Generator_CD = Generator_CD
        self.Discriminator_CD = Discriminator_CD

        # option training
        self.n_epoch = opts_training.n_epoch
        self.current_epoch = opts_training.current_epoch
        self.TrainGen  = opts_training.TrainGen
        self.TrainDisc = opts_training.TrainDisc
        self.output_results = opts_training.output_results
        
    def train(self,TrainLoader : DataLoader, ValLoader : DataLoader = None):
        best_valid_loss = np.inf

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda':
            self.Generator_CD.to(device)
            if self.Discriminator_CD is not None:
                self.Discriminator_CD.to(device)

        prev_time = time.time()

        for epoch in range(self.current_epoch, self.n_epoch):
            self.current_epoch = epoch
            for batch, (source, target) in enumerate(TrainLoader):

                real_1 = source["image"].to(device)
                real_2 = target["image"].to(device)

                

                if self.TrainGen:
                    fake_2 = self.Generator_CD(real_1)
                    if self.Discriminator_CD is not None:
                        # backward loss G
                        self.Generator_CD.optimizer.zero_grad()
                        self.Generator_CD.compute_loss(self.Discriminator_CD, real_1, real_2, fake_2)
                        self.Generator_CD.update()
                    else:
                        #train generator only mode
                        self.Generator_CD.optimizer.zero_grad()
                        self.Generator_CD.loss = self.Generator_CD.sup_loss(real_2, fake_2)
                        self.Generator_CD.update()

                if self.TrainDisc:
                    if self.TrainGen:
                        self.Discriminator_CD.optimizer.zero_grad()
                        self.Discriminator_CD.compute_loss(real_1, real_2, fake_2.detach())
                        self.Discriminator_CD.update()
                    else: #discriminator only
                        with torch.no_grad():
                            fake_2 = self.Generator_CD(real_1)
                        print("Discriminator only")
                        # backward loss D
                        self.Discriminator_CD.optimizer.zero_grad()
                        self.Discriminator_CD.compute_loss(real_1, real_2, fake_2.detach())
                        self.Discriminator_CD.update()

                # Determine approximate time left
                batches_done = epoch * len(TrainLoader) + batch
                batches_left = self.n_epoch * len(TrainLoader) - batches_done
                time_left = datetime.timedelta(
                    seconds= batches_left * (time.time() - prev_time))
                prev_time = time.time()
                

                if self.Discriminator_CD is not None:
                    if self.TrainGen & self.TrainDisc:
                        # Print log
                        sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f D_fake %f D_real %f] "
                        "[G loss: %f, pixel: %f, adv: %f] ETA: %s"
                        % (
                            epoch + 1,
                            self.n_epoch,
                            batch,
                            len(TrainLoader),
                            self.Discriminator_CD.loss.item(),
                            self.Discriminator_CD.loss_fake.item(),
                            self.Discriminator_CD.loss_real.item(),
                            self.Generator_CD.loss.item(),
                            self.Generator_CD.loss_pixel.item(),
                            self.Generator_CD.loss_gan.item(),
                            time_left,
                        )
                        )
                    elif ~self.TrainGen & self.TrainDisc:
                        sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f D_fake %f D_real %f] "
                        " ETA: %s"
                        % (
                            epoch + 1,
                            self.n_epoch,
                            batch,
                            len(TrainLoader),
                            self.Discriminator_CD.loss.item(),
                            self.Discriminator_CD.loss_fake.item(),
                            self.Discriminator_CD.loss_real.item(),
                            time_left,
                        )
                        )

                else: #Generator only
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] "
                        "[G loss: %f ] ETA: %s"
                        % (
                            epoch + 1,
                            self.n_epoch,
                            batch,
                            len(TrainLoader),
                            self.Generator_CD.loss.item(),
                            time_left,
                        )
                        )
            
            filename = os.path.join(self.output_results, 'training.tsv')

            if self.Discriminator_CD is not None:
                if self.TrainGen & self.TrainDisc:
                    columns = ['epoch', 'batch', 'loss_discriminator', 'loss_generator', 'loss_pixel', 'loss_GAN']
                    row = np.array(
                    [epoch + 1, batch, self.Discriminator_CD.loss.item(), self.Generator_CD.loss.item(),
                    self.Generator_CD.loss_pixel.item(),
                    self.Generator_CD.loss_gan.item()]
                    ).reshape(1, -1)

                    loss_test = self.Generator_CD.loss_pixel.item()
                elif ~self.TrainGen & self.TrainDisc:
                    columns = ['epoch', 'batch', 'loss_discriminator']
                    row = np.array(
                    [epoch + 1, batch, self.Discriminator_CD.loss.item()]
                    ).reshape(1, -1)
                    loss_test = self.Discriminator_CD.loss.item()
    
            else:
                columns = ['epoch', 'batch', 'loss_pixel']
                row = np.array(
                [epoch + 1, batch, self.Generator_CD.loss.item()]
                ).reshape(1, -1)

                loss_test = self.Generator_CD.loss.item()
            
            row_df = pd.DataFrame(row, columns=columns)
            self.write_tsv(filename, row_df)

            if ValLoader is not None:
                self.eval(ValLoader, device= device)

            loss_is_best = loss_test < best_valid_loss
            if loss_is_best:
                best_valid_loss = loss_test
                self.save_models(best_loss = True )
            else:
                self.save_models()

    def eval(self, ValLoader : DataLoader, device):
        
        filename = os.path.join(self.output_results, 'validation.tsv')

        loss_pixel = 0
        loss_gan = 0
        loss_discriminator = 0
        for batch, (source, target) in enumerate(ValLoader):

            real_1 = source["image"].to(device)
            real_2 = target["image"].to(device)
            with torch.no_grad():
                fake_2 = self.Generator_CD(real_1)
                if self.Discriminator_CD is not None:
                        loss_discriminator += self.Discriminator_CD.compute_loss(real_1,real_2,fake_2, eval=True)
                        if self.TrainGen & self.TrainDisc:

                            loss_gan += self.Discriminator_CD.comput_loss_GAN_g(real_1, fake_2)
                            loss_pixel += self.Generator_CD.sup_loss(real_2,fake_2)
                else:
                    loss_pixel += self.Generator_CD.sup_loss(real_2,fake_2)

        loss_discriminator = loss_discriminator / len(ValLoader)
        loss_gan = loss_gan / len(ValLoader)
        loss_pixel = loss_pixel / len(ValLoader)
                  
        if self.Discriminator_CD is not None:
                if self.TrainGen & self.TrainDisc:

                    columns = ['epoch', 'batch', 'loss_gen_gan', 'loss_pixel','loss_discriminator']
                    row = np.array(
                    [self.current_epoch + 1, batch, loss_gan.item(),
                    loss_pixel.item(), loss_discriminator.item()]
                    ).reshape(1, -1)

                elif ~self.TrainGen & self.TrainDisc:
                    columns = ['epoch', 'batch', 'loss_discriminator']
                    row = np.array(
                    [self.current_epoch + 1, batch, loss_discriminator.item()]
                    ).reshape(1, -1)

        else:
            columns = ['epoch', 'batch', 'loss_pixel']
            row = np.array(
            [self.current_epoch + 1, batch, loss_pixel.item()]
            ).reshape(1, -1)
        
        row_df = pd.DataFrame(row, columns=columns)
        self.write_tsv(filename, row_df)
    
    def write_tsv(self, filename : Path, write_df : pd.DataFrame):

        with open(filename, 'a') as f:
            write_df.to_csv(f, header= True, index = False, sep="\t")

    def save_models(self, best_loss : bool = False ):
        import os
        import shutil
        import json

        if self.Discriminator_CD is not None:
                    if self.TrainGen & self.TrainDisc:
                        gen_path = os.path.join(self.output_results,"checkpoint","gan","generator")
                        os.makedirs(gen_path, exist_ok=True)
                        self.Generator_CD.save(os.path.join(gen_path,"modelCD.pth.tar"))

                        disc_path = os.path.join(self.output_results,"checkpoint","gan","discriminator")
                        os.makedirs(disc_path, exist_ok=True)
                        self.Discriminator_CD.save(os.path.join(disc_path,"modelCD.pth.tar"))

                    elif ~self.TrainGen & self.TrainDisc:

                        disc_path = os.path.join(self.output_results,"checkpoint","gan","discriminator")
                        os.makedirs(disc_path, exist_ok=True)
                        self.Discriminator_CD.save(os.path.join(disc_path,"modelCD.pth.tar"))
        else:
            gen_path = os.path.join(self.output_results,"checkpoint","gan","generator")
            os.makedirs(gen_path, exist_ok=True)
            self.Generator_CD.save(os.path.join(gen_path,"modelCD.pth.tar"))

        save_dict = {
            "current_epoch" : self.current_epoch,
        }

        with open(os.path.join(self.output_results, "checkpoint", 'training_params.json'), 'w') as fp:
            json.dump(save_dict, fp, indent=2)
        
        if best_loss:
            best_loss_path = os.path.join(self.output_results, "best-loss")
            if not os.path.exists(best_loss_path):
                os.makedirs(best_loss_path)
            shutil.copytree(os.path.join(self.output_results,"checkpoint"), best_loss_path, dirs_exist_ok=True)