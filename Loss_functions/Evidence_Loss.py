import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((alpha.shape)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def Dice(outputs, targets):
    a = outputs * targets * 2
    a = torch.sum(a)
    b = outputs + targets
    b = torch.sum(b)
    smooth = 1e-5
    dice = (a + smooth) / (b + smooth)
    return dice

class EvidentialLoss(nn.Module):
    def __init__(self, channels=1, alpha=0.5, beta=0.5, epochs=50):
        super(EvidentialLoss, self).__init__()
        self.channels = channels
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

    def forward(self, outputs, targets, epoch):

        targets = targets.permute(0, 2, 3, 1)

        targets = targets.long()
        targets = targets.squeeze(3)
        y_one_hot = F.one_hot(targets, 2)
        y_one_hot = y_one_hot.permute(0, 3, 1, 2)

        e = outputs
        alpha = e + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = 2 / S
        P = alpha / S

        limit = - 0.5 * torch.mean(y_one_hot * (1 - P) * torch.log(u))

        loss = torch.sum(y_one_hot * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

        annealing_coef = min(1, epoch * 10 / self.epochs)
        alp = e * (1 - y_one_hot) + 1
        L_KL = annealing_coef * KL(alp, 2)

        # P = torch.argmax(outputs,dim=1,keepdim=True).squeeze(1)

        # L_Dice = (1 - Dice(P, y_one_hot)) * 2

        loss = torch.mean(loss)
        L_KL = torch.mean(L_KL)

        # SAEL
        return loss + L_KL + limit

        # EDL
        # return loss + L_KL