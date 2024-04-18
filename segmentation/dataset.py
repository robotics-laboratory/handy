import os
import cv2
import albumentations as A
import lightning as L
import numpy as np
import random
import torch
import torchvision.transforms as T

import torch.utils
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def get_non_empty_mask_crop(image, mask, size=(256, 256)):

    non_empty_col = np.argwhere(np.any(mask, axis=0)).ravel()
    non_empty_row = np.argwhere(np.any(mask, axis=1)).ravel()

    x_min, x_max = non_empty_row[0], non_empty_row[-1]
    y_min, y_max = non_empty_col[0], non_empty_col[-1]

    if x_max - size[0] + 1 >= x_min or y_max - size[1] + 1 >= y_min:
        x = random.randint(0, image.shape[0] - size[0])
        y = random.randint(0, image.shape[1] - size[1])
    else:
        x = random.randint(max(x_max - size[0] + 1, 0), x_min)
        y = random.randint(max(y_max - size[1] + 1, 0), y_min)

    return image[x:x+size[0], y:y+size[1]], mask[x:x+size[0], y:y+size[1]]


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomGamma(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0)
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0)
    ])


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.clip(mask, 0, 1)
        image, mask = get_non_empty_mask_crop(image, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask


class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, data_dir, size=(256, 256), batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.size = size
        self.batch_size = batch_size
    
    def setup(self, stage):
        self.train_dataset = SegmentationDataset(
            os.path.join(self.data_dir, 'train', 'image_bgr'),
            os.path.join(self.data_dir, 'train', 'masks'),
            size=self.size,
            transform=get_train_transform()
        )
        self.val_dataset = SegmentationDataset(
            os.path.join(self.data_dir, 'val', 'image_bgr'),
            os.path.join(self.data_dir, 'val', 'masks'),
            size=self.size,
            transform=get_val_transform()
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True
        )
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch[0].to(device), batch[1].to(device)
