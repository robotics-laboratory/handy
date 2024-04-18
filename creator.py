import cv2
import numpy as np
import sys
import os


def equalise_hist(img):
    if len(img.shape) == 3:
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    elif len(img.shape) == 2:
        img = cv2.equalizeHist(img)
    else:
        raise ValueError("Invalid number of channels")

    return img

def read_and_convert(file_path, output_folder, equal_hist=False, num_images=0, width=1920, height=1200):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    equal_dir = output_folder+"_equalised"
    if equal_hist and not os.path.exists(equal_dir):
        os.makedirs(equal_dir)


    if num_images == 0:
        num_images = os.path.getsize(file_path) // 1920 // 1200
        print(num_images)
    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        for i in range(num_images):
            # Read the image data
            img_data = f.read(width * height)

            # Convert the data to a NumPy array and reshape it to the correct dimensions
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))

            # Convert the image from Bayer RGGB to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

            # Save the image
            cv2.imwrite(os.path.join(output_folder, f'image_{i}.png'), img_rgb)
            if equal_hist:
                equaled_img = equalise_hist(img_rgb)
                cv2.imwrite(os.path.join(equal_dir, f'image_{i}.png'), img_rgb)


        print(f"written {num_images}")

if __name__ == "__main__":
    # Get the file path from the command line arguments
    file_path = sys.argv[1]
    output_folder = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "--equal":
        read_and_convert(file_path, output_folder, True)
    else:
        read_and_convert(file_path, output_folder, False)
