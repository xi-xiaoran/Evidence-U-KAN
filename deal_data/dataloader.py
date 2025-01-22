import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optin
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt


class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, channels=1, resize=False, Train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.channels = channels
        self.resize = resize
        self.Train = Train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.channels == 1:
            image = image.convert("l")
        if self.resize:
            image = image.resize((300,300), Image.Resampling.LANCZOS)
            mask = mask.resize((300,300), Image.Resampling.LANCZOS)

        if self.Train:
            image, mask = self.transform(image, mask)
        else:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.where(mask > 0.5, 1.0, 0)

        return image, mask



class TransformWithAugmentation:
    """
    包含数据增强操作的变换类。
    """
    def __init__(self, resize=(256, 256), mean=0, std=1):
        """
        初始化数据增强变换。

        :param resize: 调整后的图像大小
        :param mean: 正则化均值
        :param std: 正则化标准差
        """
        self.resize = resize
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        """
        对图像和标签应用数据增强。

        :param image: 输入图像
        :param mask: 输入标签
        :return: 增强后的图像和标签
        """
        # Resize
        image = TF.resize(image, self.resize)
        mask = TF.resize(mask, self.resize)

        # Random horizontal flip (随机水平翻转)
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip (随机垂直翻转)
        if torch.rand(1).item() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (随机旋转)
        angle = torch.randint(0, 360, (1,)).item()
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)


        # Random crop (随机裁剪)
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.resize[0] - 32, self.resize[1] - 32)
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Resize back to the original size
        image = TF.resize(image, self.resize)
        mask = TF.resize(mask, self.resize)

        # Random brightness/contrast adjustment (随机调整亮度和对比度)
        if torch.rand(1).item() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=torch.rand(1).item() + 0.5)
        if torch.rand(1).item() > 0.5:
            image = TF.adjust_contrast(image, contrast_factor=torch.rand(1).item() + 0.5)

        # Random Gaussian blur (随机高斯模糊)
        if torch.rand(1).item() > 0.5:
            image = TF.gaussian_blur(image, kernel_size=3)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[self.mean], std=[self.std])
        mask = TF.to_tensor(mask)
        mask = torch.where(mask > 0.5, 1.0, 0.0)

        return image, mask

# data_transform = transforms.Compose([
#     # transforms.CenterCrop((256,256)),
#     # transforms.Resize((256,256)),
#     transforms.Resize((256,256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0], std=[1])
# ])
# data_transform = TransformWithAugmentation(
#     resize=(256, 256), mean=0, std=1
# )


def get_data_loader(images_root, masks_root, channels=3, Train=True):
    if Train == True:
        data_transform = TransformWithAugmentation(
            resize=(256, 256), mean=0, std=1
        )
    else:
        data_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1])
        ])
    dataset = MedicalImageDataset(
        images_dir=images_root,
        masks_dir=masks_root,
        transform=data_transform,
        channels=channels,
        resize=False,
        Train=Train
    )
    return dataset

