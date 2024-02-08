from augmentations import SequentialAugmentation
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import albumentations as A

def get_augmentations(train = True):
    if train:
        return SequentialAugmentation([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Resize(256, 256),
            ToTensorV2(),
        ])
    else:
        return SequentialAugmentation([
            A.Resize(256, 256),
            ToTensorV2(),
        ])