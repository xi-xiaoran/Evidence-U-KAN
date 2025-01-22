import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optin
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import seaborn as sns

from deal_data.dataloader import get_data_loader
from train_model import train_model
from Test_model import evaluate_model
from models.Unet import UNet
from monai.networks.nets import BasicUNetPlusPlus, VNet, SwinUNETR, UNETR, AttentionUnet
from models.UKAN import UKAN
from models.Evid_U_KAN import Evid_U_KAN
from models.SwinUNETR import SwinUNETR
from models.AttentionUnet import AttentionUnet
from models.UNetplusplus import BasicUNetPlusPlus
from models.EMCAD import EMCADNet
from Loss_functions.Evidence_Loss import EvidentialLoss
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


data_name = 'CVC-ClinicDB'
# data_name = 'Kvasir-SEG'

# images_root = 'data/CVC/Original'
# masks_root = 'data/CVC/Ground_Truth'
# test_root = 'data/K/images'
# GT_root = 'data/K/masks'

images_root = 'data/K/images'
masks_root = 'data/K/masks'
test_root = 'data/CVC/Original'
GT_root = 'data/CVC/Ground_Truth'
train_dataset = get_data_loader(images_root, masks_root, channels=3, Train=False)
test_dataset = get_data_loader(test_root, GT_root, channels=3, Train=False)
random_seed = 14514

num_epochs = 300
n_classes = 2
lr = 1e-4
likehood = 0.001
std = 0
model_name = 'Evid_U_KAN'
"""
'Evid_U_KAN'
'UKAN'
'EMCAD'
'SwinUNETR'
'AttentionUnet'
'UNet++'
'UNet'
"""
Train = True
Test = False
uncertainty = True
save_path = 'save_models/' + model_name + str(data_name) + '.pth'
seed_everything(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('len dataset:', len(train_dataset))
print(device)
if Train:
    print('-' * 20, 'train', '-' * 20)
    num = 0
    model = Evid_U_KAN(input_channels=3,
                     num_classes=2).to(device)
    # model = UKAN(input_channels=3,
    #              num_classes=2).to(device)
    # model = UNet(
    #             spatial_dims=2,
    #             in_channels=3,
    #             out_channels=2,
    #             channels=(32, 64, 128, 256, 512),
    #             strides=(2, 2, 2, 2)
    #         ).to(device=device)
    # model = SwinUNETR(img_size=(256,256),
    #                   in_channels=3,
    #                   out_channels=2,
    #                   use_checkpoint=True,
    #                   spatial_dims=2).to(device)
    # model = AttentionUnet(spatial_dims=2,
    #                       in_channels=3,
    #                       out_channels=2,
    #                       channels=(32, 64, 128, 256, 512),
    #                       strides=(2, 2, 2, 2)).to(device)
    # model = EMCADNet(num_classes=2).to(device)
    # model = BasicUNetPlusPlus(in_channels=3,out_channels=2,spatial_dims=2).to(device)
    criterion = EvidentialLoss(channels=n_classes, epochs=num_epochs)
    optimizer = optin.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, uncertainty,
                        num_epochs=num_epochs, fold=num, likehood=likehood)

    torch.save(model, save_path)
    # Evaluate the model
    print('-' * 20, 'test', '-' * 20)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for std in range(0, 5, 1):
        std = (std * 1.0 / 10)
        Dice, Iou, Assd, ECE, UEO, T_U, F_U, best_T_U, best_F_U, worst_T_U, worst_F_U = evaluate_model(model,
                                                                                                       test_loader,
                                                                                                       device,
                                                                                                       uncertainty,
                                                                                                       likehood, std)
        print('std', std)
        print('Dice', Dice)
        print('Iou', Iou)
        print('ASSD', Assd)
        print('ECE', ECE)
        print('UEO', UEO)
        with open('recording.txt', 'a') as file:
            file.write(f'epoch:{num_epochs}   lr:{lr}   characteristic:{model_name}   uncertainity:{uncertainty}\n')
            file.write(f'std:{std}\nDice:{Dice}\nIou:{Iou}\nASSD:{Assd}\nECE:{ECE}\nUEO:{UEO}\n')
            file.write(f'-----------------------------------------------\n\n')
        file.close()
        print('txt write completed')

        color1 = (81 / 255, 141 / 255, 219 / 255)
        color2 = (225 / 255, 156 / 255, 102 / 255)
        sns.kdeplot(T_U, shade=True, color=color1, label='Correct')
        sns.kdeplot(F_U, shade=True, color=color2, label='Incorrect')
        plt.xlim([0, 1])
        plt.legend()
        plt.xlabel('Uncertainty', color='black')
        plt.ylabel('Density', color='black')
        plt.title('SAEL', color='darkred')
        plt.savefig(f'{num} SAEL Uncertainty distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        color1 = (81 / 255, 141 / 255, 219 / 255)
        color2 = (245 / 255, 215 / 255, 163 / 255)
        sns.kdeplot(best_T_U, shade=True, color=color1, label='Correct')
        sns.kdeplot(best_F_U, shade=True, color=color2, label='Incorrect')
        plt.xlim([0, 1])
        plt.legend()
        plt.xlabel('Uncertainty', color='black')
        plt.ylabel('Density', color='black')
        plt.title('SAEL', color='darkred')
        plt.savefig(f'{num} SAEL Best Uncertainty distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        color1 = (81 / 255, 141 / 255, 219 / 255)
        color2 = (245 / 255, 215 / 255, 163 / 255)
        sns.kdeplot(worst_T_U, shade=True, color=color1, label='Correct')
        sns.kdeplot(worst_F_U, shade=True, color=color2, label='Incorrect')
        plt.xlim([0, 1])
        plt.legend()
        plt.xlabel('Uncertainty', color='black')
        plt.ylabel('Density', color='black')
        plt.title('SAEL', color='darkred')
        plt.savefig(f'{num} SAEL Worst Uncertainty distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Image saves completed')

    torch.save(model, save_path)
    print('model saves completed')


else:
    model = torch.load(save_path)

if Test:
    num = 0
    # Evaluate the model
    print('-' * 20, 'test', '-' * 20)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for std in range(0, 5, 1):
        std = (std * 1.0 / 10)
        Dice, Iou, Assd, ECE, UEO, T_U, F_U, best_T_U, best_F_U, worst_T_U, worst_F_U = evaluate_model(model,
                                                                                                       test_loader,
                                                                                                       device,
                                                                                                       uncertainty,
                                                                                                       likehood, std)
        print('std', std)
        print('Dice', Dice)
        print('Iou', Iou)
        print('ASSD', Assd)
        print('ECE', ECE)
        print('UEO', UEO)
        with open('recording.txt', 'a') as file:
            file.write(f'epoch:{num_epochs}   lr:{lr}   characteristic:{model_name}   uncertainity:{uncertainty}\n')
            file.write(f'std:{std}\nDice:{Dice}\nIou:{Iou}\nASSD:{Assd}\nECE:{ECE}\nUEO:{UEO}\n')
            file.write(f'-----------------------------------------------\n\n')
        file.close()
        print('txt write completed')

        color1 = (81 / 255, 141 / 255, 219 / 255)
        color2 = (225 / 255, 156 / 255, 102 / 255)
        sns.kdeplot(T_U, shade=True, color=color1, label='Correct')
        sns.kdeplot(F_U, shade=True, color=color2, label='Incorrect')
        plt.xlim([0, 1])
        plt.legend()
        plt.xlabel('Uncertainty', color='black')
        plt.ylabel('Density', color='black')
        plt.title('SAEL', color='darkred')
        plt.savefig(f'{num} SAEL Uncertainty distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        color1 = (81 / 255, 141 / 255, 219 / 255)
        color2 = (245 / 255, 215 / 255, 163 / 255)
        sns.kdeplot(best_T_U, shade=True, color=color1, label='Correct')
        sns.kdeplot(best_F_U, shade=True, color=color2, label='Incorrect')
        plt.xlim([0, 1])
        plt.legend()
        plt.xlabel('Uncertainty', color='black')
        plt.ylabel('Density', color='black')
        plt.title('SAEL', color='darkred')
        plt.savefig(f'{num} SAEL Best Uncertainty distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        color1 = (81 / 255, 141 / 255, 219 / 255)
        color2 = (245 / 255, 215 / 255, 163 / 255)
        sns.kdeplot(worst_T_U, shade=True, color=color1, label='Correct')
        sns.kdeplot(worst_F_U, shade=True, color=color2, label='Incorrect')
        plt.xlim([0, 1])
        plt.legend()
        plt.xlabel('Uncertainty', color='black')
        plt.ylabel('Density', color='black')
        plt.title('SAEL', color='darkred')
        plt.savefig(f'{num} SAEL Worst Uncertainty distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Image saves completed')








