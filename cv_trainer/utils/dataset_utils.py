import os
import json
import cv2
import numpy as np

def generate_bounding_boxes(directory_path, output_path):
    # Get a list of all the mask files in the directory
    mask_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

    # Create an empty list to hold the bounding boxes
    bounding_boxes = {}

    # Loop over each mask file
    for mask_file in mask_files:
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

        # Add the bounding box to the list
        bounding_boxes[mask_file] = {
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        }
        

    # Write the bounding boxes to a JSON file
    with open(output_path, 'w') as file:
        json.dump(bounding_boxes, file)
