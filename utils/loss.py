import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import os

class MetricsTracker(object):
    def __init__(self, tag, log_root):
        self.iter_Dices = []  # to store DSCs computed at each training iteration
        self.epoch = 0  # initialize epoch at 0
        self.tag = tag  # for logging different instances of MetricsTracker
        self.log_root = log_root  # specify root directory hosting logs and tracked metrics
        self.epoch_Dices = []  # store DSCs computed at each epoch
    
    def update(self, input: torch.Tensor, target: torch.Tensor, dice):
        self.iter_Dices.append(dice)  # add computed DSCs for current iteration
    
    def get_metrics(self):
        dice = np.mean(self.iter_Dices)  # compute mean DSCs stored in iter_Dices across all iterations in current epoch
        self.epoch_Dices.append(dice)  # append computed mean DSCs to epoch_Dices
        return dice
        
    def save_logs(self):
        dict = {"tag":self.tag, "epoch": self.epoch, "eval":{"dice":self.epoch_Dices}}  # create dict containing evaluation info
        item = json.dumps(dict)  # convert dict to json-formatted string 
        with open(os.path.join(self.log_root, "{}_{}_eval.json".format(self.tag, self.epoch)), "w", encoding="utf-8") as f:
            f.write(item)  # write json-formatted string to new file
        print("Eval log {}_{}_eval.json written".format(self.tag, self.epoch))
        
    def set_epoch(self, epoch):
        self.epoch = epoch  # set epoch to 0
        self.iter_Dices = []  # resets iter_Dices to empty list to prepare for next epoch
    

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, beta=1, size_mean=True) -> None:
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth  # to prevent division by 0 and encourage smooth gradients
        self.size_mean = size_mean  # to choose whether to take Dice loss mean or sum across batch
        self.beta = beta  # to adjust balance between precision and recall
        
    def forward(self, input, target):
        N = target.size()[0]  # returns size of tensor "target" on its first dimension "batch"
        smooth = self.smooth
        
        input_flat = input.view(N, -1)  # reshape input tensor into 2D (row: sample in batch; col: flattened input tensor)
        targets_flat = target.view(N, -1)  # same for target tensor, to prepare for element-wise operations
        intersection = input_flat * targets_flat  # compute intersection of flattened input & target predictions in a tensor
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)  # DSC per sample
        loss = 1 - N_dice_eff.sum() / N  # compute loss for the batch
        return loss
