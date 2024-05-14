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

    def __init__(self, datapath, transform=None, img_transform=None):
        self.datapath = (
            '/group/dl4miacourse/The_Segmenters/Data/' + datapath
        )  # the directory with the volume
        self.vol = load_h5(self.datapath + '.h5')  # get the volume
        self.vol_label = load_h5(self.datapath + '_label.h5' ) 
        self.n_slices = self.vol.shape[0]
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )

        self.loaded_imgs = [None] * self.n_slices
        self.loaded_masks = [None] * self.n_slices
        for idx in range(self.n_slices):

            image = Image.fromarray(self.vol[idx,:,:])
            mask = Image.fromarray(self.vol_label[idx,:,:])

            self.loaded_imgs[idx] = inp_transforms(image)

            self.loaded_masks[idx] = transforms.ToTensor()(mask)

    # get the total number of samples
    def __len__(self):
        return self.n_slices

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask