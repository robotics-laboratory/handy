import argparse
import json
import os
from typing import List

import cv2
import numpy as np


def get_bbox(mask: np.ndarray) -> List[float]:
    # x_min, y_min, x_max, y_max
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    bbox = list(map(float, [x1, y1, x2, y2]))
    return bbox


def read_and_convert(sources: str, output_file: str, width=1920, height=1200) -> None:
    if not all([os.path.exists(source) for source in sources]):
        raise ValueError

    common_filenames = set(os.listdir(sources[0]))
    for source in sources[1:]:
        common_filenames.intersection_update(os.listdir(source))
    common_filenames = sorted(common_filenames)

    for i in range(len(sources)):
        source = sources[i]
        json_data = {}
        for filename in common_filenames:
            path_to_file = os.path.join(source, filename)
            image = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
            bbox = get_bbox(image)
            centroid_x = (bbox[0] + bbox[2]) / 2
            centroid_y = (bbox[1] + bbox[3]) / 2
            json_data[filename] = {}
            json_data[filename]["centroid"] = [centroid_x, centroid_y]
        with open(f"{output_file}_{i + 1}.json", mode="w", encoding="utf-8") as file:
            json.dump(json_data, file)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width", help="Width of a single frame", type=int, default=1920
    )
    parser.add_argument(
        "--height", help="height of a single frame", type=int, default=1200
    )
    parser.add_argument("--sources", help="folder with masks", required=True, nargs="*")
    parser.add_argument("--export", help="just_filename_no_extension", required=True)

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    read_and_convert(args.sources, args.export, args.width, args.height)
