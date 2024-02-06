import cv2
import numpy as np
import os
import argparse


BGR_UNDISTORT_PATH = "bgr_image_undistort"
BGR_DISTORT_PATH = "bgr_image_distort"
RAW_UNDISTORT_PATH = "raw_image_undistort"


class Intrinsics:
    def __init__(self, camera_matrix, dist_coefs):
        self.camera_matrix = np.array(
            [
                [1049.749496618372, 0, 912.9238734274475],
                [0, 1028.307877419058, 523.1240537565553],
                [0, 0, 1],
            ]
        )

        self.dist_coefs = np.array([-0.1519539, 0.337941, 0.0, 0.0, -0.1424794])
        self.image_size = None
        self.mapx, self.mapy = None, None

    def undistort_image(self, image):
        cur_image_size = image.shape[:2]
        if not self.image_size:
            self.image_size = cur_image_size
        if self.image_size != cur_image_size:
            raise ValueError("Images of different sizes were provided")

        if self.mapx is None or self.mapy is None:
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coefs,
                None,
                self.camera_matrix,
                self.image_size[::-1],
                5,
            )

        undistorted_img = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_NEAREST)
        return undistorted_img

    @staticmethod
    def create_from_yaml(path_to_file):
        obj = Intrinsics(None, None)
        return obj


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

    store_undistort_bgr = args.store_undistort_bgr
    store_distort_bgr = args.store_distort_bgr
    store_undistort_raw = args.store_undistort_raw
    export_dir = args.export
    source_dir = args.source
    params_path = args.params_path
    if params_path:
        raise NotImplementedError("Reading YAML with intrinsics is not implemented. Params are hardcoded for now")

    if not any([store_distort_bgr, store_undistort_bgr, store_undistort_raw]):
        print("At least one convertation flag must be specified")

    if not os.path.isdir(export_dir):
        print("Invalid export directory provided:", export_dir)
        return
    if not os.path.isdir(source_dir):
        print("Invalid export directory provided:", source_dir)
        return

    if store_undistort_bgr:
        intrinsic_params = Intrinsics.create_from_yaml(params_path)
        if not os.path.isdir(os.path.join(export_dir, BGR_UNDISTORT_PATH)):
            os.mkdir(os.path.join(export_dir, BGR_UNDISTORT_PATH))
    if store_distort_bgr and not os.path.isdir(
        os.path.join(export_dir, BGR_DISTORT_PATH)
    ):
        os.mkdir(os.path.join(export_dir, BGR_DISTORT_PATH))
    if store_undistort_raw:
        intrinsic_params = Intrinsics.create_from_yaml(params_path)
        if not os.path.isdir(os.path.join(export_dir, RAW_UNDISTORT_PATH)):
            os.mkdir(os.path.join(export_dir, RAW_UNDISTORT_PATH))

    for filename in os.listdir(source_dir)[:10]:
        path_to_file = os.path.join(source_dir, filename)
        image = cv2.imread(path_to_file)

        if store_undistort_raw:
            undistorted_raw_img = intrinsic_params.undistort_image(image)
            path_to_save = os.path.join(export_dir, RAW_UNDISTORT_PATH, filename)
            success = cv2.imwrite(path_to_save, undistorted_raw_img)
            if not success:
                print("ERROR", path_to_save, sep="\t")

        if store_distort_bgr or store_undistort_bgr:
            bgr_image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_BayerRGGB2BGR)
            if store_distort_bgr:
                path_to_save = os.path.join(export_dir, BGR_DISTORT_PATH, filename)
                success = cv2.imwrite(path_to_save, bgr_image)
                if not success:
                    print("ERROR", path_to_save, sep="\t")
            if store_undistort_bgr:
                path_to_save = os.path.join(export_dir, BGR_UNDISTORT_PATH, filename)
                undistort_bgr_image = intrinsic_params.undistort_image(bgr_image)
                success = cv2.imwrite(path_to_save, undistort_bgr_image)
                if not success:
                    print("ERROR", path_to_save, sep="\t")

        print("OK", path_to_file, sep="\t")


if __name__ == "__main__":
    main()
