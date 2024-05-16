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


print('Starting')

patch_size = [1,512,512]
final_activation = "Sigmoid"
batch_size = 2

model = UNet(depth=4, in_channels=1, out_channels=1, final_activation=final_activation)

# set loss function
loss_function = nn.BCELoss()

# set optimizer
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# set metrics
dice = DiceCoefficient()
n_epochs = 40

modelnum = '010'

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

print('Starting to load datasets')
train_dataset = AngioDataset('train',patch_size=patch_size)
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