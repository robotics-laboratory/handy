from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os

class SegmentationDataset(Dataset):
    TEST_SIZE = 0.2  # Define the test size

    def __init__(self, image_dir: str, mask_dir: str, is_train: bool = True, transforms = None, random_seed: int = 42):
        super().__init__()
        self.image_dir = image_dir  # Directory for images
        self.mask_dir = mask_dir  # Directory for masks
        self.transforms = transforms  # Transformations to be applied

        # Get all png files from the mask directory
        files = [file for file in sorted(os.listdir(mask_dir)) if file.endswith(".png")]

        # Split the files into train and test sets
        train_files, test_files = train_test_split(files, random_state=random_seed, test_size=self.TEST_SIZE)

        # Use train files if is_train is True, else use test files
        self.image_names = train_files if is_train else test_files

    def __len__(self):
        # Return the total number of images
        return len(self.image_names)
    
    def __getitem__(self, index: int):
        # Get the paths for the image and its corresponding mask
        image_name = os.path.join(self.image_dir, self.image_names[index])
        mask_name = os.path.join(self.mask_dir, self.image_names[index])
    
        # Read the image and convert it to RGB
        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    
        # Read the mask image and convert it to binary format
        mask_img = cv2.imread(mask_name)
        mask = np.where(mask_img.sum(axis=2) > 0, 1.0, 0.)

        # If transforms are defined, apply them to the image and mask
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Return the image and mask
        return image, mask