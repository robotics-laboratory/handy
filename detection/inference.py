import argparse 
import torch
import os
import random
import cv2
import numpy as np

from model import get_model
from albumentations.pytorch import ToTensorV2

CLASSES = ["background", "ball"]
COLORS = [[0, 0, 0], [255, 0, 0]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--size', type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--threshold', type=float)

    args = parser.parse_args()
    model = get_model(backbone_name=args.backbone, size=args.size)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    width = args.size
    height = args.size

    all_files = os.listdir(args.data_dir)
    image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg'))]

    selected_images = sorted(image_files)

    os.makedirs(args.result_dir, exist_ok=True)

    for i in range(len(selected_images)):

        image = cv2.imread(os.path.join(args.data_dir, selected_images[i]))
        orig_image = image.copy()
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image_input = torch.tensor(image_input, dtype=torch.float)
        image_input = torch.unsqueeze(image_input, 0)

        image_input = image_input.to(device)

        outputs = model(image_input)

        outputs = [{k: v for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()

            boxes = boxes[scores >= args.threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]

                xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
                ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
                xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
                ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
                cv2.rectangle(orig_image,
                            (xmin, ymin),
                            (xmax, ymax),
                            color[::-1], 
                            3)
                cv2.putText(orig_image, 
                            class_name, 
                            (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            color[::-1], 
                            2, 
                            lineType=cv2.LINE_AA)

            cv2.imwrite(os.path.join(args.result_dir, selected_images[i]), orig_image)
