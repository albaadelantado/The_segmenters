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




from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)




class DiceLoss(nn.Module):
    """
    This layer will simply compute the dice coefficient and then negate
    it with an optional offset.
    We support an optional offset because it is common to have 0 as
    the optimal loss. Since the optimal dice coefficient is 1, it is
    convenient to get 1 - dice_coefficient as our loss.

    You could leave off the offset and simply have -1 as your optimal loss.
    """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(offset), requires_grad=False)
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y):
        coefficient = self.dice_coefficient(x, y)
        return self.offset - coefficient