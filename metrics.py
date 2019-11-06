#!/usr/bin/env python
import torch.nn as nn

def dice_coeff(preds, targets):
    """Compute the dice_coeff

    Params
    ======
    preds   (torch.tensor) with shape (B * C * H * W)
    targets (torch.tensor) with shape (B * C * H * W)   
    """
    n = preds.size(0)

    smooth = 1.0

    preds   = preds.view(n, -1)
    targets = targets.view(n, -1)        

    intersection = (preds * targets).sum(1)
    
    loss = 2. * (intersection + smooth) / (preds.sum(1) + targets.sum(1) + smooth)

    return loss.mean().item()

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):        
        super().__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, preds, targets):        
        # BCE loss
        bce_loss = self.bce_loss(preds, targets)

        smooth = 1.
        intersection = (preds * targets).sum()
        dice_loss = 2. * (intersection + smooth) / (preds.sum() + targets.sum() + smooth)

        return bce_loss + (1 - dice_loss)


class MetricTracker(object):
    """
    Computes and stores the average and the current values
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum +=  val * n
        self.count += n
        self.avg = self.sum / self.count

