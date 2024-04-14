import os
import json
import cv2
import numpy as np
from tqdm import tqdm


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