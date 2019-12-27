import torch
import torch.nn as nn
import torch.functional as F

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.BCE = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        #******************** another method ***************************
        # focal loss
        # loss_pi0 = torch.abs(torch.sigmoid(pi0[...,4]) - tconf) ** gamma * BCE(pi0[..., 4], tconf)
        # lconf += (k * 64) * torch.mean(loss_pi0)
        #***************************************************************

        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss