from albumentations import DualTransform, random_crop
import random
import numpy as np

# Function to find the first and last row of a numpy matrix that includes at least one nonzero element
def find_bounding_nonzero_rows(matrix):
    nonzero_rows = np.where(matrix.any(axis=1))[0]  # Find the indices of rows with at least one nonzero element
    first_row_index, last_row_index = nonzero_rows[0], nonzero_rows[-1]  # Get the first and last row index
    return first_row_index, last_row_index

# Function to find the first and last column of a numpy matrix that includes at least one nonzero element
def find_bounding_nonzero_columns(matrix):
    nonzero_columns = np.where(matrix.any(axis=0))[0]  # Find the indices of columns with at least one nonzero element
    first_column_index, last_column_index = nonzero_columns[0], nonzero_columns[-1]  # Get the first and last column index
    return first_column_index, last_column_index

class MaskSafeRandomCrop(DualTransform):
    """
    Crop a random part of the input without loss of not null elements of mask.
    Args:
        erosion_rate (float): erosion rate applied on input image height before crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    def __init__(self, erosion_rate=0.0, p=1.0):
        super(MaskSafeRandomCrop, self).__init__(p)
        self.erosion_rate = erosion_rate
    
    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, **params):
        # Apply random crop on the image
        return random_crop(img, crop_height, crop_width, h_start, w_start)
    
    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]  # Get image height and width

        # Find bounding nonzero columns and rows in the mask
        x, x2 = find_bounding_nonzero_columns(params["mask"])
        y, y2 = find_bounding_nonzero_rows(params["mask"])

        # Normalize x's and y's by the width and height
        x, x2 = x / img_w, x2 / img_w
        y, y2 = y / img_h, y2 / img_h

        # Find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by

        # Calculate crop height and width
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)

        # Calculate start points for cropping
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)

        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

    @property
    def targets_as_params(self):
        # Return the targets as parameters
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        # Return the initial arguments names
        return ("erosion_rate",)