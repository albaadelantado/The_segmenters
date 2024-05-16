import torch
import torch.nn as nn
import numpy as np


# Sorensen Dice Coefficient implemented in torch
# the coefficient takes values in two discrete arrays
# with values in {0, 1}, and produces a score in [0, 1]
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b can be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = torch.sum(prediction*target)
        union = torch.sum(prediction)+torch.sum(target)
        return 2 * intersection / union.clamp(min=self.eps)

class IoUCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b can be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = torch.sum(prediction*target)
        union = torch.sum(prediction)+torch.sum(target)
        return intersection / (union-intersection+self.eps)
    

class IntersectionOverUnion():

    def __init__(self, prediction, target, eps=1e-6):
        self.prediction = prediction
        self.target = target
        self.eps = eps
    
    def forward(self):
        zero_one_pred = np.where(self.prediction>0, 1, 0)
        zero_one_targ = np.where(self.target>0, 1, 0)
        intersection = np.sum(zero_one_pred*zero_one_targ)
        union = np.sum(zero_one_pred)+np.sum(zero_one_targ)
        return intersection / (union - intersection + self.eps)

class DiceIndex():

    def __init__(self, prediction, target, eps=1e-6):
        self.prediction = prediction
        self.target = target
        self.eps = eps
    
    def forward(self):
        zero_one_pred = np.where(self.prediction>0, 1, 0)
        zero_one_targ = np.where(self.target>0, 1, 0)
        intersection = np.sum(zero_one_pred*zero_one_targ)
        union = np.sum(zero_one_pred)+np.sum(zero_one_targ)
        return (2*intersection) / (union + self.eps)


