from typing import List, Callable, Dict, Any
from torch import Tensor
import numpy as np
from albumentations.core.transforms_interface import DualTransform, BasicTransform
from .base import AugmentationBase

class SequentialAugmentation(AugmentationBase):
    """
    Class for performing a sequence of augmentations.
    """
    def __init__(self, augmentation_list: List[Callable]):
        """
        Initialize with a list of augmentation functions. It supports albumentations transforms, torchvision transform and custom trunsforms.
        Transformations must only be callable.
        """
        self.augmentation_list = augmentation_list

    def __call__(self, **data) -> Dict[str, Any]:
        """
        Apply the sequence of augmentations to the input data.
        """
        # Albumentations accepts only list of bboxes
        if "bbox" in data:
            data["bboxes"] = [data["bbox"]]
            data.pop("bbox")

        # Get the height and width of the image
        height, width = data["image"].shape[:2]

        # Albumentations uses normalized coordinates for top-left and bottom-right points
        if "bboxes" in data:
            for i in range(len(data["bboxes"])):
                data["bboxes"][i] = [data["bboxes"][i][0] / width, data["bboxes"][i][1] / height, data["bboxes"][i][2] / width, data["bboxes"][i][3] / height]

        # Apply each augmentation in the list to the data
        for augmentation in self.augmentation_list:
            if isinstance(augmentation, DualTransform) or isinstance(augmentation, BasicTransform):
                data.update(augmentation(**data))
            else:
                data["image"] = augmentation(data["image"])

        # If the image is a Tensor, get the height and width from the second and third dimensions
        # If not, get the height and width from the first two dimensions
        if isinstance(data["image"], Tensor):
            height, width = data["image"].shape[1:3]
        else:
            height, width = data["image"].shape[:2]

        # If bboxes are in data, scale them by the width and height of the image
        if "bboxes" in data:
            for i in range(len(data["bboxes"])):
                data["bboxes"][i] = np.array([int(data["bboxes"][i][0] * width), int(data["bboxes"][i][1] * height), int(data["bboxes"][i][2] * width), int(data["bboxes"][i][3] * height)])

        return data  

    def __repr__(self) -> str:
        """
        Return a string representation of the augmentation list.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.augmentation_list:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string