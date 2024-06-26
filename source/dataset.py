import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *


class AngioDataset(Dataset):
    """A PyTorch dataset to load angio volumes and labels"""

    def __init__(self, name, patch_size = [1,128,128], transform=None, img_transform=None):
        self.name = name
        self.path = (
            '/group/dl4miacourse/The_Segmenters/Data/' + name
        )  # the directory with the volume
        self.vol = load_h5(self.path + '.h5')  # get the volume
        self.vol_label = load_h5(self.path + '_label.h5' ) 

        self.patch_size = patch_size
        self.n_patch_per_dim = get_n_patch_per_dim(self.vol.shape, self.patch_size) 
        self.transform = transform  # transformations to apply to both inputs and targets


        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )


    # get the total number of samples
    def __len__(self):
        return np.prod(self.n_patch_per_dim)

    # fetch the training sample given its index
    def __getitem__(self, idx):

        c = True
        while  c:
            # patch_idx is 3d, idx is linear
            patch_idx = np.unravel_index(idx, self.n_patch_per_dim)
            mask_patch = get_patch(self.vol_label, patch_idx, self.patch_size).squeeze()

            c = (np.sum(mask_patch) < (0.0001 * np.prod(mask_patch.shape))) and self.name == 'train'
            #print(np.sum(mask_patch) < (0.0001 * np.prod(mask_patch.shape)))
            idx = np.random.randint(0,np.prod(self.n_patch_per_dim))
            

        #image = Image.fromarray(img_patch.squeeze())
        #mask = Image.fromarray(mask_patch.squeeze())

        img_patch = get_patch(self.vol, patch_idx, self.patch_size).squeeze()
        #image = self.inp_transforms(img_patch)
        img_patch = minmaxnorm(img_patch)
        image = torch.tensor(img_patch[np.newaxis,...].astype(np.float32))

        mask = torch.tensor(mask_patch[np.newaxis,...].astype(np.uint8))

        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform[0](image)


        if self.name == 'test':
            _, npslice = get_patch(self.vol, patch_idx, self.patch_size, return_slice = True)

            return image, mask, npslice
        else:
            return image, mask