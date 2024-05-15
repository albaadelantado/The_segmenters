from dataset import AngioDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet

from utils import *
from test import *

checkpoint_path = '/group/dl4miacourse/The_Segmenters/Checkpoints'

checkpoint_key = '001'
save_path = '/group/dl4miacourse/The_Segmenters/Predictions/' + checkpoint_key

patch_size = [1,512,512]

model = UNet(depth=4, in_channels=1, out_channels=1, final_activation=None)
model = load_checkpoint(model, checkpoint_path, optimizer=None, key='checkpoint' + checkpoint_key)

test_dataset = AngioDataset('test',patch_size=patch_size)
test_loader = DataLoader(test_dataset, batch_size=1)
vol = test_repatch(model, test_loader, saving_path = save_path)


 