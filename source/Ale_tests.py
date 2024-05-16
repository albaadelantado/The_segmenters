#Import modules and packages
from dataset import AngioDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from utils import *
from test import *

#Import test dataset
patch_dim = 512
patch_size = [1,patch_dim,patch_dim]
test_dataset = AngioDataset('test',patch_size=patch_size)
test_loader = DataLoader(test_dataset, batch_size=1)

#Indicate key
my_key = 'Ale_212'

#Indicate checkpoint path
checkpoint_path = '/group/dl4miacourse/The_Segmenters/Checkpoints'

#Load model
final_activation = "Sigmoid"
model = UNet(depth=4, in_channels=1, out_channels=1, final_activation=final_activation)
model = load_checkpoint(model, checkpoint_path, optimizer=None, key="checkpoint"+my_key)

#Test and save results
save_name = "prediction_"+my_key+".h5"
save_directory = '/group/dl4miacourse/The_Segmenters/Predictions/' + save_name
#print(save_directory)
vol = test_repatch(model, test_loader, saving_path = save_directory)


