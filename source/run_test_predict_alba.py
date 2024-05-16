from dataset import AngioDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet

from utils import load_checkpoint
from test import test_repatch

checkpoint_path = '/group/dl4miacourse/The_Segmenters/Checkpoints'

checkpoint_key = '101'
save_path = '/group/dl4miacourse/The_Segmenters/Predictions/' + checkpoint_key

final_activation = "Sigmoid"

patch_dim = 512
patch_size = [1,patch_dim,patch_dim]


model = UNet(depth=4, in_channels=1, out_channels=1, final_activation=final_activation)
model = load_checkpoint(model, checkpoint_path, optimizer=None, key='checkpoint' + checkpoint_key)

test_dataset = AngioDataset('test',patch_size=patch_size)
test_loader = DataLoader(test_dataset, batch_size=1)
vol = test_repatch(model, test_loader, saving_path = save_path)


 