#import matplotlib.pyplot as plt
#from functools import partial
#from itertools import product

import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F

#from torch.utils.data import Dataset
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.tensorboard import SummaryWriter

#import sklearn.metrics as metrics
#from sklearn.model_selection import train_test_split

#from imageio import imread
#from tqdm import tqdm, trange
from validation import validate
from train import train
from utils import save_checkpoint
import torch
import torch.optim as optim


def run_training(model, optimizer, metric, 
                 n_epochs, train_loader, val_loader, loss_function,
                 log_interval, logger, device,key="checkpoint", path="", lr_schedule = False):
    # Use the unet you expect to work the best!
    model = model.to(device)

    if lr_schedule:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1, patience = 2,
                                                       threshold=1e-4, threshold_mode="rel")
    # use adam optimizer
    #optimizer = torch.optim.Adam(model.parameters())

    # build the dice coefficient metric
    #metric = DiceCoefficient()

    # train for n_epochs
    # during the training inspect the
    # predictions in the tensorboard
    #n_epochs = 25
    for epoch in range(n_epochs):
        # train
        train(
            model,
            train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch=epoch,
            log_interval=log_interval,
            tb_logger=logger,
            device=device,
        )
        step = epoch * len(train_loader)
        # validate
        current_loss = validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)

        if lr_schedule:
            scheduler.step(current_loss)

        if len(path)>0:
            save_checkpoint(model, optimizer, epoch, path, key)



        


















































































































































































































