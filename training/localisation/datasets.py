import json
import os
import sklearn.model_selection
import cv2
import torch
import argparse
import numpy as np
import lightning as L
from torchvision.transforms import v2


from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image

class Denormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, img):
        img = (img * self.std + self.mean) * 255.
        img = img.astype(np.uint8)

        return img


def get_train_transform(width=320, height=192):
    return v2.Compose([
        #TODO: Add bbox safe random crop
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomVerticalFlip(p = 0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_valid_transform(width=320, height=192):
    return v2.Compose([
        v2.Resize(size=(height, width)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class LocalisationDataset(Dataset):
    def __init__(self, image_dir, annot_file, transforms, train=True):
        self.image_dir = image_dir
        self.annot_file = annot_file
        self.transforms = transforms
        images = []
        self.bboxes = json.load(open(annot_file))

        for file in os.listdir(self.image_dir):
            if file.endswith('.png'):
                images.append(file)
        
        images.sort()
        train_images, test_images = sklearn.model_selection.train_test_split(images, train_size=0.8, random_state=42)
        self.images = train_images if train else test_images
    
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)

        image = read_image(image_path)

        bboxes = self.bboxes[image_name]
        xmin = bboxes['xmin']
        ymin = bboxes['ymin']
        xmax = bboxes['xmax']
        ymax = bboxes['ymax']

        bboxes_resized = [[xmin, ymin, xmax, ymax]]

        target = {}
        boxes = tv_tensors.BoundingBoxes(bboxes_resized, format='XYXY', canvas_size=image.shape[-2:])
        if self.transforms:
            image, boxes_out = self.transforms(image, boxes)
            box = boxes_out[0]
            target['bbox_center'] = [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
            target['bbox_width'] = (box[2] - box[0] + box[3] - box[1]) / 4

        return image, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))


class DetectionDataModule(L.LightningDataModule):
    def __init__(self, data_dir, annot_file, width, height, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.annot_file = annot_file
        self.width = width
        self.height = height
        self.batch_size = batch_size
    
    def setup(self, stage):
        self.train_dataset = LocalisationDataset(self.data_dir, self.annot_file,
                                              transforms=get_train_transform(self.width, self.height), train=True)
        self.valid_dataset = LocalisationDataset(self.data_dir, self.annot_file,
                                              transforms=get_valid_transform(self.width, self.height), train=False)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, persistent_workers=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, persistent_workers=True)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        images, targets = batch
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        return images, targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Localisation Dataset Parameters')
    parser.add_argument('--image_dir', type=str, default='bgr_image_undistort', help='Path to the directory containing the images')
    parser.add_argument('--annot_file', type=str, default='datasets/annotated_12_02_23/bounding_boxes.json', help='Path to the file containing the bounding box annotations')
    parser.add_argument('--width', type=int, default=320, help='The width to which the images will be resized')
    parser.add_argument('--height', type=int, default=192, help='The height to which the images will be resized')
    parser.add_argument('--train', action='store_true', help='Flag indicating whether to train the model')

    args = parser.parse_args()

    transform = get_train_transform() if args.train else get_valid_transform()
    dataset = LocalisationDataset(args.image_dir, args.annot_file, transform, train=args.train)
    
    print(f"Number of samples in the train dataset: {len(dataset)}")

    def visualize(image, target):
        image = Denormalize()(image.permute(1, 2, 0).numpy())
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        center = target['bbox_center']
        width = target['bbox_width']
        cv2.circle(image, center, 5, (255, 0, 0), -1)
        cv2.imshow('image', image)
        cv2.waitKey(0)
    
    for i in range(5):
        image, target = dataset[i]
        visualize(image, target)
    cv2.destroyAllWindows()

