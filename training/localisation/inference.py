import argparse 
import torch
import os
import cv2
import numpy as np
import json
import albumentations as A

from tqdm import tqdm
from model import TTNetWithProb, BallLocalisation
from train import get_predicted_ball_pos

norm = A.Normalize(mean=(0.077, 0.092, 0.142), std=(0.068, 0.079, 0.108))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--checkpoint', type=str)

    n_last = 5


    args = parser.parse_args()
    model = TTNetWithProb(BallLocalisation())

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_files = os.listdir(args.data_dir)
    image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg'))]

    selected_images = sorted(image_files)

    os.makedirs(args.result_dir, exist_ok=True)

    bbox_centers = {}

    for i in tqdm(range(len(selected_images))):
        image_name = selected_images[i]

        image_num = int(image_name[-8:-4])
        image_prefix = image_name[:-8]

        resized_images = []

        for j in range(image_num, image_num - n_last, -1):

            if j < 0:
                image_name_l = f"{image_prefix}{str(0).rjust(4, '0')}.png"
            else:
                image_name_l= f"{image_prefix}{str(j).rjust(4, '0')}.png"
            
            image_path_l = os.path.join(args.data_dir, image_name_l)

            image = cv2.imread(image_path_l)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (320, 192)).astype(np.float32)
            image_resized /= 255.0

            image_resized = norm(image=image_resized)["image"]

            image_input = np.transpose(image_resized, (2, 0, 1)).astype(np.float32)
            image_input = torch.tensor(image_input, dtype=torch.float)

            resized_images.append(image_input)

        image_input = torch.cat(resized_images, 0).unsqueeze(0)

        image_input = image_input.to(device)


        with torch.no_grad():
            logit, cls = model(image_input)

        image_path = os.path.join(args.data_dir, image_name)
        orig_image = cv2.imread(image_path)

        if cls[0].argmax().item() == 1:

            coord = get_predicted_ball_pos(logit, 320)

            coord_x = coord[0][0].item()
            coord_y = coord[0][1].item()

            x, y =  coord_x * 1920 / 320, coord_y * 1200 / 192
            x, y = int(x), int(y)

            bbox_centers[image_name] = (x, y)

            # Рисование креста
            color = (0, 255, 0)  # Зеленый цвет
            thickness = 2
            size = 12  # Размер крестика

            # Рисование линии от верхнего левого угла до нижнего правого
            start_point = (x - size, y - size)
            end_point = (x + size, y + size)
            img = cv2.line(orig_image, start_point, end_point, color, thickness)

            # Рисование линии от верхнего правого угла до нижнего левого
            start_point = (x + size, y - size)
            end_point = (x - size, y + size)
            img = cv2.line(orig_image, start_point, end_point, color, thickness)
        else:
            bbox_centers[image_name] = (-1, -1)

        cv2.imwrite(os.path.join(args.result_dir, selected_images[i]), orig_image)

    with open(args.result_file, 'w') as f:
        json.dump(bbox_centers, f)