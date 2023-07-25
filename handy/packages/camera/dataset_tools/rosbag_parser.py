import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import os
import numpy as np
import argparse

def typename(topic_name, bag_topics):
        for topic_type in bag_topics:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

def topicname_to_dir(name):
    dir_name = name.replace('/', '_')
    if dir_name[0] == '_':
        dir_name = dir_name[1:]
    return dir_name

def check_create_dirs(save_dir, rosbag_path, topics, equalise=False):
    assert os.path.exists(rosbag_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for topic_name in topics:
        full_dir = os.path.join(save_dir, topicname_to_dir(topic_name))
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)
        if not equalise:
            continue

        full_dir = os.path.join(save_dir, topicname_to_dir("/increased"+topic_name))
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)

def equalise_hist_3chan(img):
    img_r = cv2.equalizeHist(img[:, :, 0]).reshape(1024, 1280, 1)
    img_g = cv2.equalizeHist(img[:, :, 1]).reshape(1024, 1280, 1)
    img_b = cv2.equalizeHist(img[:, :, 2]).reshape(1024, 1280, 1)
    res_img = np.concatenate((img_r, img_g, img_b), axis=2)
    return res_img

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory_to_save_files')
    parser.add_argument('path_to_bag_db3_file')
    parser.add_argument('topics_to_parse', nargs='*')
    parser.add_argument('--increase-exposure', action="store_true")

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    save_folder = args.directory_to_save_files
    rosbag_path = args.path_to_bag_db3_file
    req_topics = args.topics_to_parse
    req_increase = args.increase_exposure

    check_create_dirs(save_folder, rosbag_path, req_topics, req_increase)

    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions('', '')

    reader.open(storage, converter)

    bag_topics = reader.get_all_topics_and_types()
    bag_topics_names = [topic.name for topic in bag_topics]
    assert all([topic in bag_topics_names for topic in req_topics])

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        msg = deserialize_message(data, get_message(typename(topic, bag_topics)))
        if len(msg.data) == 1024 * 1280 * 3:
            image = np.array(msg.data, dtype=np.uint8).reshape(1024, 1280, 3)
        elif len(msg.data) == 1024 * 1280:
            image = np.array(msg.data, dtype=np.uint8).reshape(1024, 1280)
        else:
            raise ValueError

        frame_number = msg.header.stamp.nanosec % 100000
        joint_path = os.path.join(save_folder, topicname_to_dir(topic), f"{frame_number}.png")
        cv2.imwrite(joint_path, image)
        print("Image", joint_path, "saved successfully")
            
    if req_increase:
        for topic in req_topics:
            dir_to_read = os.path.join(save_folder, topicname_to_dir(topic))
            dir_to_save = os.path.join(save_folder, topicname_to_dir("/increased" + topic))

            for filename in os.listdir(dir_to_read):
                full_path = os.path.join(dir_to_read, filename)
                img = cv2.imread(full_path)

                if len(img.shape) == 3:
                    img = equalise_hist_3chan(img)
                elif len(img.shape) == 1:
                    img = cv2.equalizeHist(img)
                else:
                    raise ValueError
                
                full_new_path = os.path.join(dir_to_save, filename)
                cv2.imwrite(full_new_path, img)
                print("Image", full_new_path, "saved successfully")

        
