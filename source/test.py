import math
import torch
import torch.nn as nn
import numpy as np
from utils import save_h5, repatch


def test(
    model,
    loader,
    device=None,
    saving_path = ""
):
    
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    prediction_volume = np.zeros(loader.dataset.vol.shape)

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # TODO: evaluate this example with the given loss and metric
            prediction = model(x)
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            if len(saving_path)>0:
                np_prediction = prediction.to('cpu').numpy()
                prediction_volume[i*np_prediction.shape[0]:i*np_prediction.shape[0]+np_prediction.shape[0]] += np_prediction

    if len(saving_path)>0:
        save_h5(prediction_volume,saving_path, key='predictions')
    
    return prediction_volume
            


def test_repatch(
    model,
    loader,
    device=None,
    saving_path = ""
):
    
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        imgs = []
        slices = []
        for i, (x, y, sl) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            init_i, end_i = sl
            init_i = init_i.numpy().squeeze()
            end_i = end_i.numpy().squeeze()
            patch_npslice = np.s_[init_i[0]:end_i[0],init_i[1]:end_i[1],init_i[2]:end_i[2]]
            # TODO: evaluate this example with the given loss and metric
            prediction = model(x)
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)

            np_prediction = prediction.to('cpu').numpy()

            imgs.append(np_prediction)
            slices.append(patch_npslice)

        prediction_volume = repatch(imgs, slices, loader.dataset.vol.shape)
                

    if len(saving_path)>0:
        save_h5(prediction_volume,saving_path, key='predictions')
    
    return prediction_volume
            