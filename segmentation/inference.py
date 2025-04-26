import argparse
import torch
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import ndimage

from models import UNet
from dataset import get_val_transform


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def crop_square(img, x, y, size=128):
    h, w = img.shape[:2]
    half_size = size // 2

    # Проверка границ и корректировка координат, если необходимо
    x = max(half_size, min(x, w - half_size))
    y = max(half_size, min(y, h - half_size))

    # Вырезание квадрата
    return img[y-half_size:y+half_size, x-half_size:x+half_size], x-half_size, y-half_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--annot_path', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    model = UNet(3, 1, 8)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}, strict=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_files = os.listdir(args.data_dir)
    image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg'))]

    selected_images = sorted(image_files)

    data = json.load(open(args.annot_path))

    os.makedirs(args.result_dir, exist_ok=True)

    ball_centers = {}

    for i in tqdm(range(len(selected_images))):
        if data[selected_images[i]] == [-1, -1]:
            continue

        image_path = os.path.join(args.data_dir, selected_images[i])

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        croped_img, x_start, y_start = crop_square(img, *data[selected_images[i]], 128)

        img_t = get_val_transform()(image=croped_img)["image"]

        mask = model(img_t.unsqueeze(0))

        mask_np = torch.sigmoid(mask)[0, 0].detach().numpy()

        bin_mask_np = np.where(mask_np >args.threshold, 1, 0)

        centroid = ndimage.measurements.center_of_mass(mask_np)

        non_empty_col = np.argwhere(np.any(bin_mask_np, axis=0)).ravel()
        non_empty_row = np.argwhere(np.any(bin_mask_np, axis=1)).ravel()
    
        if len(non_empty_col) == 0 or len(non_empty_row) == 0:
            continue
            
        x_min, x_max = non_empty_row[0], non_empty_row[-1]
        y_min, y_max = non_empty_col[0], non_empty_col[-1]


        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        ball_centers[selected_images[i]] = {"center": [center[0] + x_start, center[1] + y_start], 
                                            "centroid": [centroid[1] + x_start, centroid[0] + y_start]}

        plt.imshow(croped_img)
        plt.imshow(mask_np, alpha=0.15)
        plt.scatter(centroid[1], centroid[0], color='red', label='Centroid')
        plt.scatter(center[1], center[0], color='blue', label='Center')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        plt.savefig(os.path.join(args.result_dir, selected_images[i]))
        plt.close()
    
    with open(args.result_file, 'w') as f:
        json.dump(ball_centers, f, cls=NpEncoder)
