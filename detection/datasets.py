import cv2
import numpy as np
import json
import os
import sklearn
import torch
import lightning as L
import albumentations as A


from tqdm import tqdm
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

def generate_bounding_boxes(directory_path, output_path):
    """
    Generate bounding boxes for the masks in the given directory and save them to the output file.

    Args:
        directory_path (str): The path to the directory containing the mask images.
        output_path (str): The path to the output file where the bounding boxes will be saved.
    """
    mask_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

    bounding_boxes = {}

    for mask_file in tqdm(mask_files):
        # Load the mask image
        mask_path = os.path.join(directory_path, mask_file)
        mask_img = cv2.imread(mask_path)
        mask = np.where(mask_img.sum(axis=2) > 0, 1, 0).astype(np.uint8)

        # Find the contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume there is only one object in the image
        contour = contours[0]

        # Get the bounding box of the object
        x, y, w, h = cv2.boundingRect(contour)

        bounding_boxes[mask_file] = {
            'xmin': int(x),
            'ymin': int(y),
            'xmax': int(x) + int(w),
            'ymax': int(y) + int(h)
        }
        
    with open(output_path, 'w') as file:
        json.dump(bounding_boxes, file)

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# Define the validation transforms.
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


class DetectionDataset(Dataset):
    """
    A PyTorch Dataset class for the detection task.

    Args:
        image_dir (str): The path to the directory containing the images.
        annot_file (str): The path to the file containing the bounding box annotations.
        width (int): The width to which the images will be resized.
        height (int): The height to which the images will be resized.
        transforms (callable, optional): Optional transforms to be applied to the images.
        train (bool): Whether to use the training or testing set.
    """
    def __init__(self, image_dir, annot_file, width, height, transforms=None, train=True):

        self.image_dir = image_dir
        self.annot_file = annot_file
        self.width = width
        self.height = height
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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        image_height, image_width, _ = image.shape

        bboxes = self.bboxes[image_name]
        xmin = bboxes['xmin']
        ymin = bboxes['ymin']
        xmax = bboxes['xmax']
        ymax = bboxes['ymax']

        xmin_resized = int(xmin * self.width / image_width)
        ymin_resized = int(ymin * self.height / image_height)
        xmax_resized = int(xmax * self.width / image_width)
        ymax_resized = int(ymax * self.height / image_height)

        bboxes_resized = [[xmin_resized, ymin_resized, xmax_resized, ymax_resized]]
        labels = [[1]]

        target = {}
        target['boxes'] = torch.as_tensor(bboxes_resized, dtype=torch.int64)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            sample = self.transforms(image=image_resized, bboxes=target['boxes'], labels=target['labels'])
            image_resized = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.int64)

        return image_resized, target
    
    def __len__(self):
        return len(self.images)


class DetectionDataModule(L.LightningDataModule):
    def __init__(self, data_dir, annot_dir, width, height, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.annot_dir = annot_dir
        self.width = width
        self.height = height
        self.batch_size = batch_size
    
    def setup(self, stage):
        self.train_dataset = DetectionDataset(self.data_dir, self.annot_dir, 
                                              self.width, self.height, 
                                              transforms=get_train_transform(), train=True)
        self.valid_dataset = DetectionDataset(self.data_dir, self.annot_dir,
                                              self.width, self.height,
                                              transforms=get_valid_transform(), train=False)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
    

if __name__ == '__main__':
    dataset = DetectionDataset('datasets/annotated_12_02_23/ball', 'datasets/annotated_12_02_23/bounding_boxes.json', 
                               300, 300, train=True)
    
    print(f"Number of samples in the train dataset: {len(dataset)}")

    def visualize(image, target):
        boxes = target['boxes'].numpy().astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
    
    for i in range(5):
        image, target = dataset[i]
        visualize(image, target)
    cv2.destroyAllWindows()
