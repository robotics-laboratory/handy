import argparse
import os

import cv2
import numpy as np
import yaml

BGR_UNDISTORT_PATH = "bgr_image_undistort"
BGR_DISTORT_PATH = "bgr_image_distort"
RAW_UNDISTORT_PATH = "raw_image_undistort"


class Intrinsics:
    def __init__(self, camera_matrix, dist_coefs, image_size):
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

    def undistort_image(self, image):
        cur_image_size = image.shape[:2]
        assert self.image_size == cur_image_size, (
            "Images of different sizes were provided: "
            + f"{self.image_size} != {cur_image_size}"
        )

        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_NEAREST)

    @staticmethod
    def create_from_yaml(path_to_file):
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


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="Directory with raw distorted images")
    parser.add_argument(
        "--export", help="Directotory to create folders and store result images"
    )
    parser.add_argument("--store-undistort-bgr", action="store_true")
    parser.add_argument("--store-distort-bgr", action="store_true")
    parser.add_argument("--store-undistort-raw", action="store_true")
    parser.add_argument("--params-path", default="")

    return parser


def main():
    intrinsic_params = None
    parser = init_parser()
    args = parser.parse_args()
    if not any([args.store_distort_bgr,
                args.store_undistort_bgr, args.store_undistort_raw]):  # fmt:skip
        print("At least one convertation flag must be specified")

    if not os.path.isdir(args.export):
        print("Invalid export directory provided:", args.export)
        return
    if not os.path.isdir(args.source):
        print("Invalid source directory provided:", args.source)
        return

    # load params if undistortion is enabled
    if args.store_undistort_bgr or args.store_undistort_raw:
        intrinsic_params = Intrinsics.create_from_yaml(args.params_path)

    # create directories if required
    if args.store_undistort_bgr and \
        not os.path.isdir(os.path.join(args.export, BGR_UNDISTORT_PATH)):  # fmt:skip
        os.mkdir(os.path.join(args.export, BGR_UNDISTORT_PATH))
    if args.store_distort_bgr and \
        not os.path.isdir(os.path.join(args.export, BGR_DISTORT_PATH)):  # fmt:skip
        os.mkdir(os.path.join(args.export, BGR_DISTORT_PATH))
    if args.store_undistort_raw and \
        not os.path.isdir(os.path.join(args.export, RAW_UNDISTORT_PATH)):  # fmt:skip
        os.mkdir(os.path.join(args.export, RAW_UNDISTORT_PATH))

    # proccess all images from source directory
    for filename in os.listdir(args.source):
        path_to_file = os.path.join(args.source, filename)
        image = cv2.imread(path_to_file)

        if args.store_undistort_raw:
            undistorted_raw_img = intrinsic_params.undistort_image(image)
            path_to_save = os.path.join(args.export, RAW_UNDISTORT_PATH, filename)
            success = cv2.imwrite(path_to_save, undistorted_raw_img)
            if not success:
                print("ERROR", path_to_save, sep="\t")

        if args.store_distort_bgr or args.store_undistort_bgr:
            bgr_image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_BayerRGGB2BGR)
            if args.store_distort_bgr:
                path_to_save = os.path.join(args.export, BGR_DISTORT_PATH, filename)
                success = cv2.imwrite(path_to_save, bgr_image)
                if not success:
                    print("ERROR", path_to_save, sep="\t")
            if args.store_undistort_bgr:
                path_to_save = os.path.join(args.export, BGR_UNDISTORT_PATH, filename)
                undistort_bgr_image = intrinsic_params.undistort_image(bgr_image)
                success = cv2.imwrite(path_to_save, undistort_bgr_image)
                if not success:
                    print("ERROR", path_to_save, sep="\t")

        print("OK", path_to_file, sep="\t")


if __name__ == "__main__":
    main()
