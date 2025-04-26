import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import tqdm
import yaml


class Intrinsics:
    def __init__(
        self, camera_matrix: np.ndarray, dist_coefs: np.ndarray, image_size: List[int]
    ) -> None:
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.image_size = image_size
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coefs,
            None,
            self.camera_matrix,
            self.image_size[::-1],  # (width, height)
            5,
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        cur_image_size = image.shape[:2]
        assert self.image_size == cur_image_size, (
            "Images of different sizes were provided: "
            + f"{self.image_size} != {cur_image_size}"
        )

        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_NEAREST)

    @staticmethod
    def create_from_yaml(path_to_file: str):
        if not os.path.isfile(path_to_file):
            raise FileNotFoundError
        with open(path_to_file, "r") as stream:
            data = yaml.safe_load(stream)
            camera_matrix = np.array(data["camera_matrix"], dtype=np.double).reshape(
                (3, 3)
            )
            dist_coefs = np.array(data["distorsion_coefs"], dtype=np.double)
            image_size = tuple(data["image_size"][::-1])  # (height, width)
            assert len(image_size) == 2
            return Intrinsics(camera_matrix, dist_coefs, image_size)


def get_bbox_from_mask(mask):
    mask = np.where(mask.sum(axis=2) > 0, 1, 0).astype(np.uint8)

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume there is only one object in the image
    contour = contours[0]

    # Get the bounding box of the object
    x, y, w, h = cv2.boundingRect(contour)

    return {
        "xmin": int(x),
        "ymin": int(y),
        "xmax": int(x) + int(w),
        "ymax": int(y) + int(h),
    }


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="Directory with raw distorted images")
    parser.add_argument(
        "--export", help="Directotory to create folders and store result images"
    )
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--store-undistort-raw", action="store_true")
    parser.add_argument("--params-path", default="")
    parser.add_argument("--annotation", default="")

    return parser


{
    "imagepath": {
        "has_ball": "",
        "bbox": [0, 0, 0, 0],
        "maskpath": "",
    }
}


def main() -> None:
    intrinsic_params = None
    parser = init_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.export):
        print("Invalid export directory provided:", args.export)
        return
    if not os.path.isdir(args.source):
        print("Invalid source directory provided:", args.source)
        return

    # loading or initialising annotation of the whole dataset of many folders
    # info to store for each image: whether is has a ball.
    # if it does then segmentation mask path and bounding box from mask
    dataset_annotation = {}
    dataset_annotation_path = os.path.join(args.export, "dataset_annotation.json")
    if os.path.isfile(dataset_annotation_path):
        dataset_annotation = json.load(open(dataset_annotation_path))

    # load params if undistortion is enabled
    if args.undistort:
        intrinsic_params = Intrinsics.create_from_yaml(args.params_path)

    subdir_name = args.source.split("/")[-1]

    # create directories if required
    rgb_export_dir = os.path.join(args.export, subdir_name, "images_rgb")
    mask_export_dir = os.path.join(args.export, subdir_name, "masks")
    raw_export_dir = os.path.join(args.export, subdir_name, "images_raw")

    os.makedirs(rgb_export_dir, exist_ok=True)
    os.makedirs(mask_export_dir, exist_ok=True)
    os.makedirs(raw_export_dir, exist_ok=True)

    file_annotations = json.load(open(args.annotation))

    # proccess all images from source directory
    for filename in tqdm.tqdm(os.listdir(args.source)[:50]):
        path_to_file = os.path.join(args.source, filename)
        image = cv2.imread(path_to_file)
        bgr_image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_BayerRGGB2BGR)
        # image_annotation_to_write = {
        image_annotation_to_write = {
            "has_ball": file_annotations[filename],
            "bbox": [0, 0, 0, 0],
            "maskpath": "",
        }

        if file_annotations[filename] == "ball":
            image_annotation_to_write["maskpath"] = os.path.join(
                args.source + "_mask", filename
            )
            mask = cv2.imread(image_annotation_to_write["maskpath"])
            image_annotation_to_write["bbox"] = get_bbox_from_mask(mask)

        dataset_annotation[
            os.path.join(args.export, subdir_name, filename)
        ] = image_annotation_to_write

        if args.undistort:
            image = intrinsic_params.undistort_image(image)
            bgr_image = intrinsic_params.undistort_image(bgr_image)

        cv2.imwrite(os.path.join(rgb_export_dir, filename), bgr_image)
        cv2.imwrite(os.path.join(raw_export_dir, filename), image)

    json.dump(dataset_annotation, open(dataset_annotation_path, mode="w"))


if __name__ == "__main__":
    main()
