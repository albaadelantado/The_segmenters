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
from test import test_repatch
import torchvision.transforms.v2 as transforms_v2
from customtransform import DiscreteRotateTransform

#Indicate trasformations
transform = None #transforms_v2.Compose([transforms_v2.RandomHorizontalFlip(0.5), transforms_v2.RandomVerticalFlip(0.5),  DiscreteRotateTransform([0,90,180,270])])
img_transform = transforms_v2.Compose([transforms_v2.GaussianBlur(3, sigma=1)]),


#Import train and validation datasets
patch_dim = 512
patch_size = [1,patch_dim,patch_dim]
train_dataset = AngioDataset('train', patch_size, img_transform=img_transform)
val_dataset = AngioDataset("val", patch_size, img_transform=img_transform)

# pass data to DataLoader
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# pass to device
device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

# set model's parameters
final_activation="Sigmoid"
model = UNet(depth=4, in_channels=1, out_channels=1, final_activation=final_activation).to(device)

# set loss function
loss_function = nn.BCELoss()

# set optimizer
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# set metrics
dice = DiceCoefficient()

# indicate key
my_key  = "Ale_213"

# set logger's parameters
logger = SummaryWriter(f"runs/{my_key}")
log_interval=1


# model's training
n_epochs = 15
checkpoint_save_path  = "/group/dl4miacourse/The_Segmenters/Checkpoints"
run_training(model, optimizer, dice, n_epochs,
             train_loader, val_loader, loss_function, log_interval, 
             logger, device=device, key="checkpoint"+my_key, path=checkpoint_save_path)


