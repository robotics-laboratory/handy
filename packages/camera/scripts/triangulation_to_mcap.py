import argparse
import os

import cv2
import numpy as np
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Quaternion
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation

SEC_MULTIPLIER = 10**9
MS_MULTIPLIER = 10**6
MCS_MULTIPLIER = 10**3
NANO_MULTIPLIER = 1


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return qx, qy, qz, qw


class CameraParameters:
    def __init__(
        self,
        image_size,
        camera_matrix,
        dist_coefs,
        camera_id,
        rotation_vector,
        translation_vector,
    ):
        self.camera_matrix = np.array(camera_matrix, dtype=float).reshape((3, 3))
        self.dist_coefs = np.array(dist_coefs, dtype=float)
        self.camera_id = camera_id
        self.image_size = image_size

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coefs,
            None,
            self.camera_matrix,
            self.image_size,
            cv2.CV_32FC1,
        )

        self.rotation_vector = np.array(rotation_vector, dtype=float)
        self.translation_vector = np.array(translation_vector, dtype=float)
        self.rotation_matrix = cv2.Rodrigues(self.rotation_vector.reshape((3, 1)))[0]

        self.static_transformation = TransformStamped()
        self.static_transformation.child_frame_id = f"camera_{self.camera_id}"
        self.static_transformation.header.frame_id = "world"

        self.static_transformation.transform.translation.x = self.translation_vector[0]
        self.static_transformation.transform.translation.y = self.translation_vector[1]
        self.static_transformation.transform.translation.z = self.translation_vector[2]

        qx, qy, qz, qw = Rotation.from_matrix(self.rotation_matrix).as_quat().tolist()

        self.static_transformation.transform.rotation.x = qx
        self.static_transformation.transform.rotation.y = qy
        self.static_transformation.transform.rotation.z = qz
        self.static_transformation.transform.rotation.w = qw

    def undistort(self, image):
        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_NEAREST)

    def publish_camera_info(self, writer, current_time):
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = f"camera_{self.camera_id}"
        camera_info_msg.header.stamp.sec = current_time % SEC_MULTIPLIER
        camera_info_msg.header.stamp.nanosec = current_time // SEC_MULTIPLIER
        camera_info_msg.height = self.image_size[0]
        camera_info_msg.width = self.image_size[1]

        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.d = self.dist_coefs.flatten().tolist()

        camera_info_msg.k = self.camera_matrix.flatten().tolist()

        camera_info_msg.p = (
            (
                self.camera_matrix
                @ np.concatenate(
                    (self.rotation_matrix, self.translation_vector.reshape((3, 1))),
                    axis=1,
                )
            )
            .flatten()
            .tolist()
        )

        writer.write(
            f"/camera_{self.camera_id}/info",
            serialize_message(camera_info_msg),
            current_time,
        )

    def publish_transform(self, writer, current_time):
        self.static_transformation.header.stamp.sec = current_time % SEC_MULTIPLIER
        self.static_transformation.header.stamp.nanosec = current_time // SEC_MULTIPLIER

        writer.write("/tf", serialize_message(self.static_transformation), current_time)


def init_writer(export_file):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=export_file, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/triangulation/ball_marker",
            type="visualization_msgs/msg/Marker",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/triangulation/table_plane",
            type="visualization_msgs/msg/Marker",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/tf",
            type="geometry_msgs/msg/TransformStamped",
            serialization_format="cdr",
        )
    )

    for i in range(2):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=f"/camera_{i + 1}/image",
                type="sensor_msgs/msg/CompressedImage",
                serialization_format="cdr",
            )
        )
        # writer.create_topic(
        #     rosbag2_py.TopicMetadata(
        #         name=f"/camera_{i + 1}/detected_ball_arrow",
        #         type="visualization_msgs/msg/Marker",
        #         serialization_format="cdr",
        #     )
        # )
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=f"/camera_{i + 1}/info",
                type="sensor_msgs/msg/CameraInfo",
                serialization_format="cdr",
            )
        )

    return writer


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask-sources", help="folder with masks", required=True, nargs="*"
    )
    parser.add_argument(
        "--rgb-sources", help="folder with masks", required=True, nargs="*"
    )
    parser.add_argument(
        "--intrinsic-params", help="yaml file with intrinsic parameters", required=True
    )
    parser.add_argument(
        "--detection-result", help="yaml with the result of detection", required=True
    )
    parser.add_argument("--export", help="some_file.mcap", required=True)

    return parser


def init_ball_marker(marker_id, current_time, position, camera_id, ttl=100):
    msg = Marker()
    msg.header.frame_id = f"camera_{camera_id}"
    msg.header.stamp.sec = current_time // SEC_MULTIPLIER
    msg.header.stamp.nanosec = current_time % SEC_MULTIPLIER
    msg.ns = "ball_markers"
    msg.id = marker_id
    msg.type = Marker.SPHERE
    msg.action = Marker.ADD
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position

    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.1  # size of the ball (1m diameter)
    msg.scale.y = 0.1
    msg.scale.z = 0.1

    msg.color.r = 1.0  # orange color
    msg.color.g = 0.5
    msg.color.b = 0.0
    msg.color.a = 1.0  # alpha (1.0 = opaque, 0.0 = transparent)

    msg.lifetime.nanosec = ttl * 10**6  # ttl = 10ms

    return msg


def simulate(writer, mask_sources, rgb_sources, filename_to_3d_points, intrinsics):
    filenames_to_publish = sorted(filename_to_3d_points.keys())
    cv_bridge_instance = CvBridge()

    current_simulation_time = 0  # in nanoseconds
    for i in range(len(filenames_to_publish))[:100]:
        filename = filenames_to_publish[i]
        for camera_idx in range(2):
            # load and segmentate image
            image = cv2.imread(os.path.join(rgb_sources[camera_idx], filename))
            mask = cv2.imread(
                os.path.join(mask_sources[camera_idx], filename), cv2.IMREAD_GRAYSCALE
            )
            if image is None or mask is None:
                print(os.path.join(rgb_sources[camera_idx], filename))
                print(os.path.join(mask_sources[camera_idx], filename))
                quit()
            assert image.shape[:2] == mask.shape
            # segmentated_image = image
            # segmentated_image[:, :, 0] = image[:, :, 0] * mask
            # segmentated_image[:, :, 1] = segmentated_image[:, :, 1] * mask
            # segmentated_image[:, :, 2] = segmentated_image[:, :, 2] * mask
            # Highlight color and intensity

            image = intrinsics[camera_idx].undistort(image)

            highlight_color = [0, 255, 255]  # Yellow color
            alpha = 0.3  # Intensity of highlight
            # Create an image with highlight color
            highlight = np.full(image.shape, highlight_color, dtype=np.uint8)
            # Bitwise-AND mask and original image
            highlighted_area = cv2.bitwise_and(highlight, highlight, mask=mask)
            # Alpha blend highlighted_area and original image
            image = cv2.addWeighted(image, 1, highlighted_area, alpha, 0)

            # prepare CompressedImages
            camera_feed_msg = cv_bridge_instance.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            camera_feed_msg.header.frame_id = f"camera_{camera_idx + 1}"
            camera_feed_msg.header.stamp.sec = current_simulation_time // SEC_MULTIPLIER
            camera_feed_msg.header.stamp.nanosec = (
                current_simulation_time % SEC_MULTIPLIER
            )

            # publish images
            writer.write(
                f"/camera_{camera_idx + 1}/image",
                serialize_message(camera_feed_msg),
                current_simulation_time,
            )
            # publish camera info and transformations
            intrinsics[camera_idx].publish_camera_info(writer, current_simulation_time)
            intrinsics[camera_idx].publish_transform(writer, current_simulation_time)

        ball_marker = init_ball_marker(
            i, current_simulation_time, filename_to_3d_points[filename], camera_idx + 1
        )
        writer.write(
            "/triangulation/ball_marker",
            serialize_message(ball_marker),
            current_simulation_time,
        )
        current_simulation_time += 15 * MS_MULTIPLIER  # 15 ms between the frames


def init_camera_info(writer, params_path, camera_ids=[1, 2]):
    intrinsics = []
    for camera_id in camera_ids:
        with open(params_path, mode="r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            intrinsics.append(
                CameraParameters(
                    data["parameters"][camera_id]["image_size"],
                    data["parameters"][camera_id]["camera_matrix"],
                    data["parameters"][camera_id]["distortion_coefs"],
                    camera_id,
                    data["parameters"][camera_id]["rotation"],
                    data["parameters"][camera_id]["translation"],
                )
            )
    print(data.keys())
    complanar_aruco_points = np.array(data["triangulated_common_points"], dtype=float)
    print(complanar_aruco_points.shape)
    centroid = np.mean(complanar_aruco_points, axis=0)
    print("centroid is", centroid)
    _, _, VT = np.linalg.svd(complanar_aruco_points[:10] - centroid, full_matrices=False)
    print("normal is", VT[-1, :])
    return intrinsics, VT[-1, :], centroid


def publish_table_plain(writer, normal, centroid):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = 100.0  # Adjust as needed
    marker.scale.y = 100.0  # Adjust as needed
    marker.scale.z = 0.01  # Thin along the z-axis
    marker.color.a = 1.0  # Don't forget to set the alpha!
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.8
    marker.pose.position.x = centroid[0]
    marker.pose.position.y = centroid[1]
    marker.pose.position.z = centroid[2]
    # You'll need to convert the normal vector to a quaternion for the pose orientation
    # This is a bit involved - you might want to use a helper function
    marker.pose.orientation = normal_to_quaternion(normal)
    writer.write("/triangulation/table_plane", serialize_message(marker), 0)

def normal_to_quaternion(normal):
    # Ensure the normal is a unit vector
    normal = normal / np.linalg.norm(normal)

    # Compute the angle between the normal and the z-axis
    angle = np.arccos(np.dot(normal, [0, 0, 1]))

    # Compute the axis of rotation
    axis = np.cross(normal, [0, 0, 1])
    axis = axis / np.linalg.norm(axis)

    # Use scipy to create a rotation and convert it to a quaternion
    rotation_vector = angle * axis
    print('++++++++++++++')
    print(rotation_vector)

    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    quat = Quaternion()
    quat.x, quat.y, quat.z, quat.w = Rotation.from_matrix(rotation_matrix).as_quat().tolist()    
    return quat


if __name__ == "__main__":
    # init all modules
    parser = init_parser()
    args = parser.parse_args()
    writer = init_writer(args.export)

    # read and publish camera info
    intrinsics, table_plane_normal, table_plain_centroid = init_camera_info(
        writer, args.intrinsic_params, [1, 2]
    )
    publish_table_plain(writer, table_plane_normal, table_plain_centroid)

    with open(args.detection_result, mode="r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    filenames_to_3d_points = dict(
        zip(data[list(data.keys())[0]]["filenames"], data["triangulated_points"])
    )

    simulate(
        writer, args.mask_sources, args.rgb_sources, filenames_to_3d_points, intrinsics
    )

    del writer
