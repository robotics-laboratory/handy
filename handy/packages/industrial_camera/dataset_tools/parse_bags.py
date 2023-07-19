import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import os
import cv2
import numpy as np

PATH = "../../rosbag2_2023_07_19-09_54_23/rosbag2_2023_07_19-09_54_23_0.db3"
SAVE_FOLDER = "save_folder_2ms"


class BagParser:
    def __init__(self, path_to_file):
        self.conn = sqlite3.connect(path_to_file)
        self.cursor = self.conn.cursor()

        self.topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of: type_of for id_of, name_of, type_of in self.topics_data}
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in self.topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in self.topics_data}

        print(self.topic_type)
        print(self.topic_msg_message)

    
    def __del__(self):
        self.conn.close()

    def get_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id))
        print(rows.rowcount, "rows in", topic_name)
        for i in range(10**6):
            row = rows.fetchone()
            if row is None:
                return None
            yield (row[0], deserialize_message(row[1], self.topic_msg_message[topic_name]))
    

def equalise_hist(dir, save_folder_name):
    if not os.path.exists(os.path.join(SAVE_FOLDER, save_folder_name)):
        os.mkdir(os.path.join(SAVE_FOLDER, save_folder_name))

    read_dir = os.path.join(SAVE_FOLDER, dir)
    for filename in os.listdir(read_dir):
        img = cv2.imread(os.path.join(read_dir, filename))

        img_r = cv2.equalizeHist(img[:, :, 0]).reshape(1024, 1280, 1)
        img_g = cv2.equalizeHist(img[:, :, 1]).reshape(1024, 1280, 1)
        img_b = cv2.equalizeHist(img[:, :, 2]).reshape(1024, 1280, 1)

        final_image = np.concatenate((img_r, img_g, img_b), axis=2)

        new_fileneme = "equalised_" + filename
        full_path = os.path.join(SAVE_FOLDER, save_folder_name, new_fileneme)
        cv2.imwrite(full_path, final_image)
        print("Image", full_path, "corrected and saved successfully")


if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

if not os.path.exists(os.path.join(SAVE_FOLDER, "raw_images")):
    os.mkdir(os.path.join(SAVE_FOLDER, "raw_images"))

if not os.path.exists(os.path.join(SAVE_FOLDER, "converted_images")):
    os.mkdir(os.path.join(SAVE_FOLDER, "converted_images"))


parser = BagParser(PATH)
bridge = CvBridge()

for msg in parser.get_messages("/camera/converted_image_1"):
    if msg is None:
        break
    image = bridge.imgmsg_to_cv2(msg[1])
    frame_number = msg[1].header.frame_id.split('_')[-1]
    filename = f"converted_image_{frame_number}.png"
    joint_path = os.path.join(SAVE_FOLDER, "converted_images", filename)

    cv2.imwrite(joint_path, image)
    print("Image", joint_path, "saved successfully")



msgs = parser.get_messages("/camera/raw_image_1")
for msg in msgs:
    image = np.array(msg[1].data, dtype=np.uint8).reshape(1024, 1280)
    frame_number = msg[1].header.frame_id.split('_')[-1]
    filename = f"raw_image_{frame_number}.png"
    joint_path = os.path.join(SAVE_FOLDER, "raw_images", filename)

    cv2.imwrite(joint_path, image)
    print("Image", joint_path, "saved successfully")


equalise_hist("converted_images", "increased_exposure_converted")
equalise_hist("raw_images", "increased_exposure_raw")