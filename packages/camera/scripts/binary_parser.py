import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def save_image(i, img_data, output_folder, width, height):
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
    cv2.imwrite(os.path.join(output_folder, f"image_{str(i).rjust(4, '0')}.png"), img)


def read_and_convert(file_path, output_folder, width=1920, height=1200):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_images = os.path.getsize(file_path) // width // height
    print(f"{num_images} images of size {height}x{width} will be written to {output_folder}")

    executor = ThreadPoolExecutor()
    with open(file_path, "rb") as f:
        for i in tqdm(range(num_images)):
            img_data = f.read(width * height)
            executor.submit(save_image, i, img_data, output_folder, width, height)


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width", help="Width of a single frame", type=int, default=1920
    )
    parser.add_argument(
        "--height", help="height of a single frame", type=int, default=1200
    )
    parser.add_argument("--source", help="Binary file.out", required=True)
    parser.add_argument(
        "--export",
        help="Directotory to store raw single-channel .png files",
        required=True,
    )

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    read_and_convert(args.source, args.export, args.width, args.height)
