import json
import os
import sklearn.model_selection
import cv2
import torch
import argparse
import numpy as np
import lightning as L
import albumentations as A


from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class Denormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, img):
        img = (img * self.std + self.mean) * 255.
        img = img.astype(np.uint8)

        return img


def get_train_transform():
    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.077, 0.092, 0.142), std=(0.068, 0.079, 0.108), max_pixel_value=1, p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    return A.ReplayCompose([
        A.Normalize(mean=(0.077, 0.092, 0.142), std=(0.068, 0.079, 0.108), max_pixel_value=1, p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

class LocalisationDataset(Dataset):
    def __init__(self, image_dir, annot_file, width, height, n_last = 5, transforms=None):

        self.image_dir = image_dir
        self.annot_file = annot_file
        self.width = width
        self.height = height
        self.n_last = n_last
        self.transforms = transforms
        self.bboxes = json.load(open(annot_file))

        images = []
        for file in os.listdir(self.image_dir):
            if file.endswith('.png') and file in self.bboxes:
                images.append(file)
        
        images.sort()
        self.images = images
    
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)


        image_num = int(image_name[-8:-4])
        image_prefix = image_name[:-8]

        resized_images = []

        bboxes = self.bboxes[image_name]
        xmin = bboxes['xmin']
        ymin = bboxes['ymin']
        xmax = bboxes['xmax']
        ymax = bboxes['ymax']

        for i in range(image_num, image_num - self.n_last, -1):

            image_name = f"{image_prefix}{str(i).rjust(4, '0')}.png"
            image_path = os.path.join(self.image_dir, image_name)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized /= 255.0

            resized_images.append(image_resized)

        
        image_height, image_width, _ = image.shape
        xmin_resized = int(xmin * self.width / image_width)
        ymin_resized = int(ymin * self.height / image_height)
        xmax_resized = int(xmax * self.width / image_width)
        ymax_resized = int(ymax * self.height / image_height)

        if ymin_resized == ymax_resized:
            if ymin_resized == 0:
                ymax_resized += 1
            else:
                ymin_resized -= 1
        if xmin_resized == xmax_resized:
            if xmin_resized == 0:
                xmax_resized += 1
            else:
                xmin_resized -= 1

        bboxes_resized = torch.as_tensor([[xmin_resized, ymin_resized, xmax_resized, ymax_resized]], dtype=torch.int64)
        labels = torch.as_tensor([1], dtype=torch.int64)

        aug_images = []

        if self.transforms:
            data = self.transforms(image=resized_images[0], bboxes=bboxes_resized, labels=labels, return_replay=True)
            bboxes_resized = data['bboxes']
            replay = data['replay']
            aug_images.append(data['image'])

            for i in range(1, self.n_last):
                data = A.ReplayCompose.replay(replay, image=resized_images[i], bboxes=bboxes_resized, labels=labels)
                aug_images.append(data['image'])
            
            resized_images = aug_images
        
        target = {}
        box = bboxes_resized[0]
        target['ball_center'] = [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
        target['ball_rad'] = (box[2] - box[0] + box[3] - box[1]) / 4

        stack = torch.cat(resized_images, dim=0)

        return stack, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images)
    return images, targets


class LocalisationDataModule(L.LightningDataModule):
    def __init__(self, data_dir, annot_file, width=320, height=192, n_last=5,  batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.annot_file = annot_file
        self.width = width
        self.height = height
        self.n_last = n_last
        self.batch_size = batch_size
    
    def setup(self, stage):
        self.train_dataset = LocalisationDataset(os.path.join(self.train_dir, 'images_rgb'), os.path.join(self.train_dir, 'boxes.json'),
                                              self.width, self.height, self.n_last,
                                              transforms=get_train_transform())
        self.valid_dataset = LocalisationDataset(os.path.join(self.val_dir, 'images_rgb'), os.path.join(self.val_dir, 'boxes.json'),
                                              self.width, self.height, self.n_last,
                                              transforms=get_valid_transform())
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, persistent_workers=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, persistent_workers=True)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        images, targets = batch
        return images.to(device), targets
