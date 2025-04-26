import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, pred_mask, target):
        pred = torch.sigmoid(pred_mask)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1.0 - (
            (2.0 * intersection + self.smooth)
            / (pred.sum() + target.sum() + self.smooth)
        )
