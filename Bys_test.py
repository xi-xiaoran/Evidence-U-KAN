import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optin
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from PIL import Image
import matplotlib.pyplot as plt
import ASSD
from ASSD import assd


def get_tp_tn_fp_fn(preds, labels):
    tn = len([(i, j) for i, j in zip(labels, preds) if i == 0 and j == 0])
    tp = len([(i, j) for i, j in zip(labels, preds) if i == 1 and j == 1])
    fn = len([(i, j) for i, j in zip(labels, preds) if i == 1 and j == 0])
    fp = len([(i, j) for i, j in zip(labels, preds) if i == 0 and j == 1])
    return tp, tn, fp, fn


def add_noise(outputs, mean, stddev, device):
    # mean = 0
    # stddev = 0.5
    noise = np.random.normal(mean, stddev, outputs.shape)
    noise = torch.from_numpy(noise).to(device)
    outputs = outputs + noise
    outputs = outputs.type(torch.float32)
    return outputs


def Dice(A, B):
    smooth = 1e-8
    I = A * B
    T = A + B
    dice = 2 * I.sum() / (T.sum() + smooth)
    return dice


def Iou(A, B):
    I = A * B
    T = A + B
    U = T - I
    smooth = 1e-8
    iou = I.sum() / (U.sum() + smooth)
    return iou

def assd_score(o, t):
    o = o.cpu().numpy()
    t = t.cpu().numpy()
    s = assd(o, t)
    return s

def ECE_score(P, GT):
    P = P.cpu()
    GT = GT.cpu()
    right1 = 1e-8
    right2 = 1e-8
    right3 = 1e-8
    right4 = 1e-8
    right5 = 1e-8
    false1 = 1e-8
    false2 = 1e-8
    false3 = 1e-8
    false4 = 1e-8
    false5 = 1e-8
    P1 = 0
    P2 = 0
    P3 = 0
    P4 = 0
    P5 = 0
    # P torch.Size([4, 2, 256, 256])
    pred_label = torch.argmax(P,dim=1,keepdim=True)
    # torch.Size([4, 1, 256, 256])
    one_hot_GT = F.one_hot(GT.squeeze(1).long())
    one_hot_GT = one_hot_GT.permute(0,3,1,2)
    # torch.Size([4, 2, 256, 256])
    P = P * one_hot_GT
    P,_ = torch.max(P,dim=1,keepdim=True)
    # torch.Size([4, 1, 256, 256])
    for a in range(P.shape[0]):
        for c in range(P.shape[2]):
            for d in range(P.shape[3]):
                now_P = P[a][0][c][d]
                right = (pred_label[a][0][c][d] == GT[a][0][c][d])
                if now_P>=0 and now_P<0.2:
                    P1 = P1 + now_P
                    if right:
                        right1 = right1 + 1
                    else:
                        false1 = false1 + 1
                elif now_P>=0.2 and now_P<0.4:
                    P2 = P2 + now_P
                    if right:
                        right2 = right2 + 1
                    else:
                        false2 = false2 + 1
                elif now_P>=0.4 and now_P<0.6:
                    P3 = P3 + now_P
                    if right:
                        right3 = right3 + 1
                    else:
                        false3 = false3 + 1
                elif now_P>=0.6 and now_P<0.8:
                    P4 = P4 + now_P
                    if right:
                        right4 = right4 + 1
                    else:
                        false4 = false4 + 1
                elif now_P>=0.8 and now_P<=1:
                    P5 = P5 + now_P
                    if right:
                        right5 = right5 + 1
                    else:
                        false5 = false5 + 1
    P1 = P1 / (right1 + false1)
    ACC1 = right1 / (right1 + false1)
    P2 = P2 / (right2 + false2)
    ACC2 = right2 / (right2 + false2)
    P3 = P3 / (right3 + false3)
    ACC3 = right3 / (right3 + false3)
    P4 = P4 / (right4 + false4)
    ACC4 = right4 / (right4 + false4)
    P5 = P5 / (right5 + false5)
    ACC5 = right5 / (right5 + false5)
    ECE = abs(P1 - ACC1) * (right1 + false1) + abs(P2 - ACC2) * (right2 + false2) + abs(P3 - ACC3) * (right3 + false3) + abs(P4 - ACC4) * (right4 + false4) + abs(P5 - ACC5) * (right5 + false5)
    ECE = ECE / (right1 + false1 + right2 + false2 + right3 + false3 + right4 + false4 + right5 + false5)

    return ECE


# Evaluate model
def evaluate_model(model, test_loader, device, uncertainty, std):
    # model.eval()
    model.train()
    running_loss = 0.0
    num = 0
    dice = 0
    iou = 0
    assd = 0
    ece = 0
    ueo = 0
    best_ueo = 0
    worst_ueo = 1
    T_U = np.array([])
    best_T_U = np.array([])
    worst_T_U = np.array([])
    F_U = np.array([])
    best_F_U = np.array([])
    worst_F_U = np.array([])
    num_samples = 50
    for inputs, labels in test_loader:
        num = num + 1
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
        # if True:
            outputs = torch.zeros(inputs.size(0), inputs.size(2), inputs.size(3), 1, num_samples)
            inputs = add_noise(inputs, mean=0, stddev=std, device=device)
            for i in range(num_samples):
                outputs[:, :, :, :, i] = model(inputs).permute(0, 2, 3, 1)
            mean_outputs = torch.mean(outputs,dim=4)
            P1 = mean_outputs.permute(0, 3, 1, 2)
            P0 = 1 - mean_outputs
            P0 = P0.permute(0, 3, 1, 2)
            P = torch.cat((P0,P1),dim=1)
            # B 2 H W
            pred = torch.where(mean_outputs>0.5,1,0).permute(0, 3, 1, 2).to(device)
            # B 1 H W
            u = -torch.sum(P * torch.log(P + 1e-5), dim=1, keepdim=True)
            # B 1 H W
            u = u.squeeze(0)
            a = inputs.squeeze(0).squeeze(0).permute(1, 2, 0)
            b = pred.squeeze(0).squeeze(0)
            c = labels.squeeze(0).squeeze(0)
            d = u.squeeze(0).squeeze(0)
            a = a.cpu().numpy()
            b = b.cpu().numpy()
            c = c.cpu().numpy()
            d = d.cpu().detach().numpy()
            dice = dice + Dice(pred, labels)
            iou = iou + Iou(pred, labels)
            assd = assd + assd_score(pred, labels)
            ece = ece + ECE_score(P, labels)
            wucha = pred.long() ^ labels.long()
            e = wucha.squeeze(0).squeeze(0)
            e = e.cpu().numpy()
            U = d
            ueo = ueo + Dice(U, e)
            dice1 = Dice(pred, labels)
            dice1 = dice1.cpu().numpy()
            iou1 = Iou(pred, labels)
            iou1 = iou1.cpu().numpy()
            assd1 = assd_score(pred, labels)
            ece1 = ECE_score(P, labels)
            ece1 = ece1.cpu().numpy()
            ueo1 = Dice(U, e)
            diff = e
            U = U.ravel()
            U = U / 0.7
            diff = diff.ravel()
            new_T_U = U[diff == 0]
            new_F_U = U[diff == 1]
            if len(T_U) <= 100000:
                T_U = np.concatenate((T_U, new_T_U))
            if len(F_U) <= 100000:
                F_U = np.concatenate((F_U, new_F_U))
            if ueo1 > best_ueo:
                best_ueo = ueo1
                best_T_U = new_T_U
                best_F_U = new_F_U
            if ueo1 < worst_ueo:
                worst_ueo = ueo1
                worst_T_U = new_T_U
                worst_F_U = new_F_U
            print('Dice:', dice1, 'Iou:', iou1, 'ASSD:', assd1, 'ECE:', ece1, 'UEO:', ueo1)
            plt.subplot(1, 5, 1)
            plt.axis('off')
            plt.imshow(a)
            plt.title('input')
            plt.subplot(1, 5, 2)
            plt.axis('off')
            plt.imshow(b)
            plt.title('pred')
            plt.subplot(1, 5, 3)
            plt.axis('off')
            plt.imshow(c)
            plt.title('GT')
            plt.subplot(1, 5, 4)
            plt.axis('off')
            plt.imshow(d)
            plt.title('Uncertainity')
            plt.subplot(1, 5, 5)
            plt.axis('off')
            plt.imshow(e)
            plt.title('loss')
            plt.savefig(f'see/{num}.png')
            plt.close()
    dice = dice / len(test_loader)
    iou = iou / len(test_loader)
    assd = assd / len(test_loader)
    ece = ece / len(test_loader)
    ueo = ueo / len(test_loader)
    return dice, iou, assd, ece, ueo, T_U, F_U, best_T_U, best_F_U, worst_T_U, worst_F_U
