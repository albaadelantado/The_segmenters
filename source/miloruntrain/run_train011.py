# import required modules
from dataset import AngioDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from run_training import run_training
from torch.utils.tensorboard import SummaryWriter
from metric import DiceCoefficient
from utils import *
from test import *

import torchvision.transforms.v2 as transforms_v2
from customtransform import DiscreteRotateTransform


print('Starting')

patch_size = [1,512,512]
final_activation = "Sigmoid"
batch_size = 1

model = UNet(depth=4, in_channels=1, out_channels=1, final_activation=final_activation)

# set loss function
loss_function = nn.BCELoss()

# set optimizer
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# set metrics
dice = DiceCoefficient()

n_epochs = 40

modelnum = '011'

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

transform = transforms_v2.Compose([transforms_v2.RandomHorizontalFlip(0.5), transforms_v2.RandomVerticalFlip(0.5),  DiscreteRotateTransform([0,90,180,270])])#, transforms_v2.RandomAffine(degrees = 0, shear = 0, translate = [0.1,0.1], scale = [0.1,0.1])])
img_transform = None # transforms_v2.Compose([transforms_v2.GaussianBlur(3, sigma=1)]),

print('Starting to load datasets')
train_dataset = AngioDataset('train',patch_size=patch_size, transform = transform, img_transform = img_transform)
val_dataset = AngioDataset('val',patch_size=patch_size)


# pass data to DataLoader
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
print('Data loaded')

# pass to device

model = model.to(device)


# set logger's parameters
logger = SummaryWriter(f'runs/{modelnum}')
log_interval=1

# model's training


checkpoint_path = '/group/dl4miacourse/The_Segmenters/Checkpoints'
run_training(model, optimizer, dice, n_epochs,
             train_loader, val_loader, loss_function, log_interval, logger, device=device, 
             path = '/group/dl4miacourse/The_Segmenters/Checkpoints',
             key = 'checkpoint' + modelnum)