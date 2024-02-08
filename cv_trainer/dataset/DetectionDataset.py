from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import json
import torch
from random import shuffle

class DetectionDataset(Dataset):
    TEST_SIZE = 0.2  # Define the test size

    def __init__(self, image_dir: str, total_dir: str, bbox_json: str, is_train: bool = True, transforms = None, random_seed: int = 42, negatives_p: float = 0.5, **kwargs):
        super().__init__()
        self.image_dir = image_dir  # Directory for images with ball
        self.total_dir = total_dir  # Directory for total images
        self.bboxes = json.load(open(bbox_json))  # Load bounding box data from json
        self.transforms = transforms  # Transformations to be applied

        # Get all keys (file names) from the bounding box dictionary
        files = self.bboxes.keys()

        # Split the files into train and test sets
        train_files, test_files = train_test_split(list(files), random_state=random_seed, test_size=self.TEST_SIZE)

        # Get the negative files (those not in image_dir)
        negative_files = [file for file in sorted(os.listdir(total_dir)) if file.endswith(".png") and file not in os.listdir(image_dir)]

        # Split the negative files into train and test sets
        train_neg, test_neg = train_test_split(negative_files, random_state=random_seed, test_size=self.TEST_SIZE)

        # Use train files if is_train is True, else use test files
        self.image_names = train_files if is_train else test_files

        # Extend the image names with negative files
        self.image_names.extend((train_neg if is_train else test_neg)[:int(len(self.image_names) * negatives_p)])
        shuffle(self.image_names)

    def __len__(self):
        # Return the total number of images
        return len(self.image_names)
    
    def __getitem__(self, index: int):
        # Check if the image has a bounding box
        if self.image_names[index] in self.bboxes:
            # If yes, get the image from image_dir and get its bounding box
            image_name = os.path.join(self.image_dir, self.image_names[index])
            bbox = self.bboxes[self.image_names[index]]
            mark = 1
        else:
            # If no, get the image from total_dir and set bounding box to 0
            image_name = os.path.join(self.total_dir, self.image_names[index])
            bbox = {
                'x': 0,
                'y': 0,
                'width': 0,
                'height': 0
            }
            mark = 0

        # Read the image and convert it to RGB
        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

        bbox = [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']]

        # If transforms are defined, apply them to the image and bounding box
        if self.transforms is not None:
            augmented = self.transforms(image=image, bbox=bbox)
            image = augmented["image"]
            bbox = augmented["bboxes"][0]

        # Return the image, bounding box, and mark
        return {
            'image': image,
            'bbox': bbox,
            'mark': mark
        }
 

def collate_fn(batch):
    images = [item['image'] for item in batch]
    marks = [item['mark'] for item in batch]
    bboxes = [item['bbox'] for item in batch]

    images = torch.stack(images, axis=0)
    bboxes = torch.tensor(bboxes)  
    marks = torch.tensor(marks)

    return {
        'image': images,
        'bbox': bboxes,
        'mark': marks
    }
