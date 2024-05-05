import argparse
import os

import cv2
import numpy as np
import yaml


def get_bbox(image):
    a = np.where(image != 0)
    bbox = [
        float(np.min(a[0])),
        float(np.min(a[1])),
        float(np.max(a[0])),
        float(np.max(a[1])),
    ]
    return bbox


def read_and_convert(sources, output_file, width=1920, height=1200):
    if not all([os.path.exists(source) for source in sources]):
        raise ValueError

    common_filenames = set(os.listdir(sources[0]))
    for source in sources[1:]:
        common_filenames.intersection_update(os.listdir(source))
    common_filenames = sorted(common_filenames)

    yaml_data = {}
    for source in sources:
        res_data = []
        for filename in common_filenames:
            path_to_file = os.path.join(source, filename)
            image = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
            res_data.append(get_bbox(image))

        yaml_data[source] = {"filenames": common_filenames, "bounding_boxes": res_data}

    with open(output_file, mode="w", encoding="utf-8") as file:
        yaml.dump(yaml_data, file, default_flow_style=None)


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width", help="Width of a single frame", type=int, default=1920
    )
    parser.add_argument(
        "--height", help="height of a single frame", type=int, default=1200
    )
    parser.add_argument("--sources", help="folder with masks", required=True, nargs="*")
    parser.add_argument("--export", help="some_file.yaml", required=True)

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    read_and_convert(args.sources, args.export, args.width, args.height)
