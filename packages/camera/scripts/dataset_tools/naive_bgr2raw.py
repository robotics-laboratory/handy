import cv2
import numpy as np
import argparse
import os


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern", default="RGGB", help="Bayer pattern, default='RGGB'"
    )
    parser.add_argument(
        "--source",
        help="where to get the original images, can be a directory or a single file",
    )
    parser.add_argument(
        "--export",
        help="directory where to save the converted images of the same filenames",
    )

    return parser


def init_bayer(pattern):
    RGB_SYMB = "RGB"
    if not set(pattern).issubset(set(RGB_SYMB)):
        raise ValueError("only R G B letters are allowed in Bayer pattern argument")
    if len(pattern) != 4:
        raise ValueError("4 letters for Bayer pattern must be provided")

    res = [
        [RGB_SYMB.index(pattern[0]), RGB_SYMB.index(pattern[1])],
        [RGB_SYMB.index(pattern[2]), RGB_SYMB.index(pattern[3])],
    ]

    return res


def rgb_to_bayer(rgb_data, bayer_pattern):
    height, width, _ = rgb_data.shape
    bayer_data = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            bayer_data[i, j] = rgb_data[i, j, bayer_pattern[i % 2][j % 2]]

    return bayer_data


def proccess_image(path_to_read, path_to_write, pattern):
    image = cv2.imread(path_to_read)
    if image is None:
        return False
    bayer_data = rgb_to_bayer(image, pattern)
    return cv2.imwrite(path_to_write, bayer_data)


def main():
    parser = init_parser()
    args = parser.parse_args()
    source = args.source
    export_dir = args.export
    bayer_pattern = init_bayer(args.pattern.upper())

    if not os.path.isdir(export_dir):
        raise ValueError("export directory does not exist")

    if os.path.isfile(source):
        filename = os.path.split(source)[-1]
        path_to_write = os.path.join(export_dir, filename)
        if not proccess_image(source, path_to_write, bayer_pattern):
            print(path_to_write, "====", "FAILED", sep="\t")
            return
        print(path_to_write, "====", "OK", sep="\t")
        return

    for filename in os.listdir(source):
        path_to_read = os.path.join(source, filename)
        if not os.path.isfile(path_to_read):
            continue
        path_to_write = os.path.join(export_dir, filename)
        if not proccess_image(path_to_read, path_to_write, bayer_pattern):
            print(path_to_write, "====", "FAILED", sep="\t")
            continue
        print(path_to_write, "====", "OK", sep="\t")


if __name__ == "__main__":
    main()
