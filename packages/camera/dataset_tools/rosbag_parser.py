import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import os
import numpy as np
import argparse
from cv_bridge import CvBridge

FRAME_SIZE = (1024, 1280)


def get_type_by_topic(bag_topics):
    type_by_topic = {}
    for topic in bag_topics:
        type_by_topic[topic.name] = topic.type
    return type_by_topic


def topic_to_dir(name):
    dir_name = name.replace("/", "_")
    if dir_name[0] == "_":
        dir_name = dir_name[1:]
    return dir_name


def create_dirs(save_dir, topics, equalise=False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for topic_name in topics:
        full_dir = os.path.join(save_dir, topic_to_dir(topic_name))
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)

        if equalise:
            full_dir = os.path.join(save_dir, topic_to_dir("/increased" + topic_name))
            if not os.path.exists(full_dir):
                os.mkdir(full_dir)


def equalise_hist(img):
    if len(img.shape) == 3:
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    elif len(img.shape) == 2:
        img = cv2.equalizeHist(img)
    else:
        raise ValueError("Invalid number of channels")

    return img


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs='*')
    parser.add_argument("--from-bag")
    parser.add_argument("--export")
    parser.add_argument("--increase-exposure", action="store_true")

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    save_folder = args.export
    rosbag_path = args.from_bag
    req_topics = args.topics
    req_increase = args.increase_exposure
    print(rosbag_path)
    if not os.path.exists(rosbag_path):
        print("Bag file not found")
        print("Aborting")
        return

    create_dirs(save_folder, req_topics, req_increase)

    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions("", "")
    bridge = CvBridge()

    reader.open(storage, converter)

    bag_topics = reader.get_all_topics_and_types()
    type_by_topic = get_type_by_topic(bag_topics)

    bag_topics_names = set([topic.name for topic in bag_topics])
    absent_topics = set(req_topics) - bag_topics_names
    if absent_topics:
        print("[ERROR] Not found all topics:")
        print("\n".join(absent_topics))
        print("Aborting")
        return

    while reader.has_next():
        topic, data, msg_timestamp = reader.read_next()
        if topic not in req_topics:
            continue

        msg = deserialize_message(data, get_message(type_by_topic[topic]))
        image = bridge.compressed_imgmsg_to_cv2(msg)

        filename = f"{msg.header.stamp.sec}{msg.header.stamp.nanosec}.png"
        joint_path = os.path.join(save_folder, topic_to_dir(topic), filename)
        success = cv2.imwrite(joint_path, image)
        print(f"{'OK' if success else 'ERROR'}   {joint_path}")

        if req_increase:
            dir_to_save = os.path.join(save_folder, topic_to_dir("/increased" + topic))

            img = equalise_hist(image)

            full_new_path = os.path.join(dir_to_save, filename)
            success = cv2.imwrite(full_new_path, img)
            print(f"{'OK' if success else 'ERROR'}   {full_new_path}")


if __name__ == "__main__":
    main()
