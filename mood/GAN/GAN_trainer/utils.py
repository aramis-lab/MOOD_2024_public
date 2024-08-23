
#%%
import torch
import pandas as pd # type: ignore
import os 
from pathlib import Path
#%%

def index_split(
        label_path: os.PathLike,
        split_path: os.PathLike,
) -> list:

    df_label = pd.read_csv(label_path, sep="\t")
    df_label = df_label[['participant_id','session_id']]
    df_split = pd.read_csv(split_path, sep="\t")

    merged = pd.merge(df_label, df_split, on=['participant_id','session_id'], how = 'left', indicator=True)
    index_pick = merged.loc[merged['_merge'] == 'both']
    index_pick = list(index_pick.index)

    return index_pick



def extract_patch_tensor(
    image_tensor: torch.Tensor,
    patch_size: int,
    stride_size: int,
    patch_index: int,
    patches_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """Extracts a single patch from image_tensor"""

    if patches_tensor is None:
        # if it is a batch of tensor (N,C,D,H,W) (N: batch size)
        if len(image_tensor.size()) == 5:
             batch_size = image_tensor.size()[0]
             patches_tensor = (
            image_tensor.unfold(2, patch_size, stride_size)
            .unfold(3, patch_size, stride_size)
            .unfold(4, patch_size, stride_size)
            .contiguous()
        )
             patches_tensor = patches_tensor.view(batch_size,-1, patch_size, patch_size, patch_size)
             return patches_tensor[:,patch_index, ...].unsqueeze_(1).clone()
        # the return is a tensor dim [ batch_size, 1, patch_size, patch_size, patch_size]
        else:
            patches_tensor = (
                image_tensor.unfold(1, patch_size, stride_size)
                .unfold(2, patch_size, stride_size)
                .unfold(3, patch_size, stride_size)
                .contiguous()
            )
            patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)
            return patches_tensor[patch_index, ...].unsqueeze_(0).clone()
        # the dimension of patches_tensor is [1, patch_num1, patch_num2, patch_n
    else:
         return patches_tensor[patch_index, ...].unsqueeze_(0).clone()

def save_checkpoint(state, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                   best_loss='best_loss'):
    import torch
    import os
    import shutil

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(state, os.path.join(checkpoint_dir, filename))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        if not os.path.exists(best_loss_path):
            os.makedirs(best_loss_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))

#%% 

def get_data_caps(caps_sub_directory):
    subjects = pd.Series(os.listdir(caps_sub_directory))
    subjects = subjects[subjects.str.contains('sub-')]
    data_caps_df = pd.DataFrame(columns=['participant_id', 'session_id'])

    for subject in subjects:
        subject_path = os.path.join(caps_sub_directory, subject)
        if os.path.isdir(subject_path):
            sessions = pd.Series(os.listdir(subject_path))
            sessions = sessions[sessions.str.contains('ses-')]
            df1 = pd.DataFrame({'participant_id': len(sessions)*[subject],
                                'session_id': sessions})
            data_caps_df = pd.concat([data_caps_df,df1])

    mask = data_caps_df['session_id'].str.contains('ses-')
    return data_caps_df[mask]

#%%
if __name__ == "__main__":
    bids_directory = Path('/network/lustre/iss02/aramis/datasets/adni/bids/BIDS/')

    subjects = pd.Series(os.listdir(bids_directory))
    # subjects = subjects[subjects.str.contains('sub-')]
    bids_df = pd.DataFrame(columns=['participant_id', 'session_id'])

    for subject in subjects:
        subject_path = os.path.join(bids_directory, subject)
        if os.path.isdir(subject_path):
            sessions = pd.Series(os.listdir(subject_path))
            sessions = sessions[sessions.str.contains('ses-')]
            df1 = pd.DataFrame({'participant_id': len(sessions)*[subject],
                                'session_id': sessions})
            bids_df = pd.concat([bids_df,df1])

#%%

def check_labels_caps(
        label_path: os.PathLike,
        caps_directory: os.PathLike,
) -> pd.DataFrame:

    df_label = pd.read_csv(label_path, sep="\t")
    df_label = df_label[['participant_id','session_id']]

    df_caps = get_data_caps(os.path.join(caps_directory,'subjects'))
    
    merged = pd.merge(df_caps, df_label , on=['participant_id','session_id'], how = 'both')

    return merged

#%%
if __name__ == "__main__":
    import time
    from copy import copy

    caps_directory = "/network/lustre/iss02/aramis/datasets/adni/caps/caps_pet_uniform"
    label_tsv = 'ADNI_GAN/tsv_file/labels.tsv'

    df_label = pd.read_csv(label_tsv, sep="\t")
    # df_label = df_label[['participant_id','session_id']]

    df_caps = get_data_caps(os.path.join(caps_directory,'subjects'))

    start_time = time.time()
    merged = pd.merge(df_caps, df_label , on=['participant_id','session_id'], how = 'inner')
    print("--- %s seconds ---" % (time.time() - start_time))

    # merged.to_csv('ADNI_GAN/tsv_file/labels_final2.tsv', sep="\t",index=False )

    start_time = time.time()
    df_caps.set_index(["participant_id", "session_id"], inplace=True)
    bids_copy_df = copy(df_caps)
    for subject, session in df_caps.index.values:
                subject_qc_df = df_label[
                    (df_label.participant_id == subject)
                    & (df_label.session_id == session)
                ]
                if len(subject_qc_df) != 1:
                            df_caps.drop((subject, session), inplace=True)

    print("--- %s seconds ---" % (time.time() - start_time))

class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())
    
import torch.nn.functional as F
class ResizeInterpolation(object):
  """ Interpolation  """
   
  def __init__(self, size: tuple, mode : str = 'trilinear', align_corners : bool = False):
      self.size = size
      self.mode = mode
      self.align_corners = align_corners

  def __call__(self,image : torch.Tensor) -> torch.Tensor:
    if len(image.size()) == 3:
        image = image.unsqueeze(0)
    image  = image.unsqueeze(0)
    image = F.interpolate(image, size= self.size, mode= self.mode, align_corners= self.align_corners).squeeze(0)
    return image

class ResidualMean(object):
     
    def __init__(self,mean_path: Path):
          self.mean = torch.load(mean_path)
        
    def __call__(self, image : torch.Tensor) -> torch.Tensor:
        return  image - self.mean

class ThresholdNormalization(object):

    def __call__(self,image : torch.Tensor) -> torch.Tensor:
        values, counts = torch.unique(torch.round(image, decimals = 2), return_counts = True)
        threshold = values[counts.argmax()]
        image[image< threshold] = threshold
        return image
from skimage.filters import sobel
class Sobel(object):

    def __call__(self,image):
        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0)


class MEANandIMAGE_sobel(object):
    
    def __init__(self, mean_tensor : torch.Tensor):
        image = mean_tensor.squeeze().numpy()
        self.mean_tensor = torch.from_numpy(sobel(image)).unsqueeze(0)
    
    def __call__(self, image : torch.Tensor):

        image = image.squeeze().numpy()
        return torch.from_numpy(sobel(image)).unsqueeze(0) + self.mean_tensor
    
#%% Learning rate scheduler 

# adapted from google research : https://github.com/google-research/google-research/blob/master/adversarial_nets_lr_scheduler/demo.ipynb (tensorflow)
from torch.optim.lr_scheduler import LRScheduler, _enable_get_lr_call
import warnings
EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

class GAP_scheduler(LRScheduler):

    def __init__(self,optimizer, last_epoch =-1, verbose="deprecated" ,ideal_loss = 0.4, x_min = 0.1*0.4,
                  x_max = 0.1*0.4, h_min = 0.1, f_max = 2.0):
        self.ideal_loss = ideal_loss
        self.x_min = x_min
        self.x_max = x_max
        self.h_min = h_min
        self.f_max = f_max

        self.smoothing_loss = ideal_loss
        self.factor = 1.0

        super().__init__(optimizer, last_epoch, verbose)
    
    def get_factor(self, loss):
        self.smoothing_loss = 0.95 * self.smoothing_loss + 0.05 * loss
        print(self.smoothing_loss)
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        
        def cond(pred, true_branch, false_branch):
            if pred:
                return true_branch()
            else:
                return false_branch()
            
        x = torch.abs(self.smoothing_loss-self.ideal_loss)
        f_x = torch.clip(torch.pow(self.f_max, x/self.x_max), 1.0, self.f_max)
        h_x = torch.clip(torch.pow(self.h_min, x/self.x_min), self.h_min, 1.0)
        factor = cond(self.smoothing_loss > self.ideal_loss, lambda: f_x, lambda: h_x)
        self.factor = factor.item()
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
            
        return [group['initial_lr'] * self.factor
                for group in self.optimizer.param_groups] 
    
    def step(self,loss= None , epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if loss is not None:
            self.get_factor(loss)

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()
                

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

if __name__ == "__main__": 
    ideal_loss: float = 0.4
    x_min: float = 0.1*0.4
    x_max: float = 0.1*0.4
    h_min: float = 0.1
    f_max: float = 2.0

    lrate = torch.tensor(0.0001)
    discriminator_loss = torch.tensor([0.450294,0.370137,0.408395, 0.375484, 0.305861])
    generator_loss = torch.tensor(0.6987331509590149) 


    from mood.GAN.GAN_trainer.model import Discriminator64
    from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

    discriminator = Discriminator64()
    optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))

    optimizer1 = torch.optim.Adam(
            discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))

    
    scheduler = GAP_scheduler(optimizer=optimizer,
                              last_epoch=-1,
                              ideal_loss=0.5,
                              x_min= 0.1*0.5,
                              x_max= 0.1*0.5)
    smoothed_disc_loss = torch.tensor(0.5)
    ideal_loss = 0.5
    lr_hand = optimizer1.param_groups[0]["lr"]
    for epoch in range(len(discriminator_loss)):
        loss = discriminator_loss[epoch]
        smoothed_disc_loss = 0.95 * smoothed_disc_loss + 0.05 * loss
        print("smoothed_disc_loss",smoothed_disc_loss)
        optimizer.step()
        scheduler.step(loss = loss)

        lr_hand =  lr_hand * lr_scheduler(smoothed_disc_loss, ideal_loss, x_min, x_max, h_min=0.1, f_max=2.0)


        print("scheduler learning rate : ", scheduler.get_last_lr())
        print("learning rates by hand : ", lr_hand)
        # print(optimizer.param_groups[)
