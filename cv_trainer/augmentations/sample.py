from augmentations import SequentialAugmentation
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

def get_augmentations(train = True):
    if train:
        return SequentialAugmentation([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            Resize(256, 256),
            ToTensorV2(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return SequentialAugmentation([
            Resize(256, 256),
            ToTensorV2(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])