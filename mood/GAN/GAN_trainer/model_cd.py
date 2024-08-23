#%%
import torch
import torch.nn as nn
import sys
#import sys
sys.path.append('../')
from pathlib import Path
# Generic model 

class ModelEvolve(nn.Module):

    def __init__(self, architecture : nn.Module , optimizer, scheduler):

        super().__init__()
        self.architecture = architecture
        self.optimizer = optimizer
        self.loss = None
        self.scheduler = scheduler

    def forward(self, input):
        return self.architecture(input)

    def update(self):

        self.loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step() 

    def set_requires_grad(self, requires_grad : bool =False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for param in self.architecture.parameters():
            param.requires_grad = requires_grad

    def save(self,path : Path):
        """ Save the architecture, optimizer, loss if need to restart
        """
        state = {"architecture": self.architecture.state_dict(),
                 "optimizer" : self.optimizer.state_dict(),
                 "loss" : self.loss}
        torch.save(state, path)
        
    def load(self,path : Path):

        state = torch.load(path, map_location="cpu")

        self.architecture.load_state_dict(state["architecture"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.loss = state["loss"]


class ConvPatchD(ModelEvolve):

    def __init__(self, discriminator : nn.Module, optimizer, loss_GAN, scheduler = None):

        super().__init__(architecture= discriminator,
                        optimizer= optimizer,
                        scheduler= scheduler)

        self.loss_GAN = loss_GAN
        self.valid = None
        self.fake = None

    def compute_loss(self, real_1, real_2, fake_2, eval = False):

        self.fake = (0.3) * torch.rand((real_2.size(0), 1, 14, 14, 14), 
                                    device= fake_2.device, requires_grad=False)

        if self.valid is None:
            self.valid = (1 - 0.7) * torch.rand((real_2.size(0), 1, 1, 1, 1),
                                    device = real_2.device, requires_grad=False) + 0.7
        
        pred_real = self.forward(real_2, real_1)  
        loss_real =  self.loss_GAN(pred_real, self.valid)

        pred_fake = self.forward(fake_2, real_1)
        loss_fake = self.loss_GAN(pred_fake, self.fake)

        self.valid = None
        if eval:
            return  0.5* (loss_fake + loss_real)
        else:
            self.loss = 0.5* (loss_fake + loss_real)

    def comput_loss_GAN_g(self, real_1 : torch.Tensor, fake_2: torch.Tensor):
        self.valid = (1 - 0.7) * torch.rand((fake_2.size(0), 1, 14, 14, 14),
                                    device = fake_2.device, requires_grad=False) + 0.7

        pred_fake = self.forward(fake_2, real_1)
        loss_GAN =  self.loss_GAN(pred_fake, self.valid)

        return loss_GAN 
from mood.GAN.GAN_trainer.utils import extract_patch_tensor

class DecomposedD(ModelEvolve):
    
    def __init__(self, discriminator : nn.Module, optimizer , loss_GAN ,  scheduler = None):

        super().__init__(architecture= discriminator,
                        optimizer= optimizer,
                        scheduler= scheduler)

        self.loss_GAN = loss_GAN

        self.fake = None
        self.valid = None

    def forward(self, real_1, to_comp):
        return self.architecture(real_1, to_comp)

    def compute_loss(self, real_1, real_2, fake_2, eval = False):

        loss_real = 0
        loss_fake = 0

        self.fake = (0.3) * torch.rand((real_2.size(0), 1, 1, 1, 1), 
                                    device= fake_2.device, requires_grad=False)
        
        if self.valid is None:
            self.valid = (1 - 0.7) * torch.rand((real_2.size(0), 1, 1, 1, 1),
                                    device = real_2.device, requires_grad=False) + 0.7

        for index_ in range(8):
           
            real_1_patch = extract_patch_tensor(real_1, 64, 50, index_)
            real_2_patch = extract_patch_tensor(real_1, 64, 50, index_)
            fake_2_patch = extract_patch_tensor(fake_2, 64, 50, index_)

            pred_real = self.forward(real_2_patch, real_1_patch)  
            loss_real = loss_real + self.loss_GAN(pred_real, self.valid)

            pred_fake = self.forward(fake_2_patch.detach(), real_1_patch.detach())
            loss_fake = loss_fake + self.loss_GAN(pred_fake, self.fake)

        self.loss_real = loss_real / 8
        self.loss_fake = loss_fake / 8

        self.valid = None
        if eval:
            return  0.5* (self.loss_fake + self.loss_real)
        else:
            self.loss = 0.5* (self.loss_fake + self.loss_real)

    def comput_loss_GAN_g(self, real_1 : torch.Tensor, fake_2: torch.Tensor):
        self.valid = (1 - 0.7) * torch.rand((fake_2.size(0), 1, 1, 1, 1),
                                    device = fake_2.device, requires_grad=False) + 0.7

        loss_GAN = 0
        for index_ in range(8):

            real_1_patch = extract_patch_tensor(real_1, 64, 50, index_)
            fake_2_patch = extract_patch_tensor(fake_2, 64, 50, index_)

            pred_fake = self.forward(fake_2_patch, real_1_patch)
            loss_GAN = loss_GAN + self.loss_GAN(pred_fake, self.valid)

        return loss_GAN / 8 
        

class Generator(ModelEvolve):

    def __init__(self, generator, optimizer,  sup_loss, scheduler, lambda_g = 1.0,
    lmabda_p = 100.):

        super().__init__(architecture= generator,
                        optimizer= optimizer,
                        scheduler= scheduler)


        self.sup_loss = sup_loss
        #lambda (gan, pixelwise)
        self.lambda_g = lambda_g
        self.lambda_p = lmabda_p

        # attribute to keep 
        self.loss_gan = None
        self.loss_pixel = None

    def compute_loss(self, discriminator : ModelEvolve, real_1 : torch.Tensor, real_2 : torch.Tensor, fake_2 : torch.Tensor):

        self.loss_gan = discriminator.comput_loss_GAN_g(real_1,fake_2)
        self.loss_pixel = self.sup_loss(real_2, fake_2)
        self.loss = self.lambda_g * self.loss_gan + self.lambda_p * self.loss_pixel
        

  #%%  

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)