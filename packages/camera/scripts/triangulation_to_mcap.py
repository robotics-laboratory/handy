import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, TransformStamped
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import ImageMarker, Marker

SEC_MULTIPLIER = 10**9
MS_MULTIPLIER = 10**6
MCS_MULTIPLIER = 10**3
NANO_MULTIPLIER = 1

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
FPS_LATENCY_MS = 15

last_marker_id = 0
last_table_marker_id = None


def get_new_marker_id() -> int:
    global last_marker_id
    last_marker_id += 1
    return last_marker_id


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
        print(
            "angles are",
            Rotation.from_matrix(self.rotation_matrix).as_euler("zyx") / 3.14 * 180,
        )

        self.static_transformation.transform.rotation.x = qx
        self.static_transformation.transform.rotation.y = qy
        self.static_transformation.transform.rotation.z = qz
        self.static_transformation.transform.rotation.w = qw

    def undistort(self, image: cv2.Mat) -> cv2.Mat:
        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_NEAREST)

    def publish_camera_info(
        self, writer: rosbag2_py.SequentialWriter, current_time: int
    ) -> None:
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = f"camera_{self.camera_id}"
        camera_info_msg.header.stamp.sec = current_time % SEC_MULTIPLIER
        camera_info_msg.header.stamp.nanosec = current_time // SEC_MULTIPLIER
        camera_info_msg.height = self.image_size[1]
        camera_info_msg.width = self.image_size[0]

        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.d = self.dist_coefs.flatten().tolist()

        camera_info_msg.k = self.camera_matrix.flatten().tolist()
        camera_info_msg.p = self.camera_matrix.flatten().tolist()

        writer.write(
            f"/camera_{self.camera_id}/info",
            serialize_message(camera_info_msg),
            current_time,
        )

    def publish_transform(
        self, writer: rosbag2_py.SequentialWriter, current_time: int
    ) -> None:
        self.static_transformation.header.stamp.sec = current_time % SEC_MULTIPLIER
        self.static_transformation.header.stamp.nanosec = current_time // SEC_MULTIPLIER

        writer.write("/tf", serialize_message(self.static_transformation), current_time)


def init_writer(export_file: str) -> rosbag2_py.SequentialWriter:
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
            name="/triangulation/ball_table_projection",
            type="visualization_msgs/msg/Marker",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/triangulation/trajectory",
            type="visualization_msgs/msg/Marker",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/triangulation/intersection_points",
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
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=f"/camera_{i + 1}/ball_center",
                type="visualization_msgs/msg/ImageMarker",
                serialization_format="cdr",
            )
        )
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=f"/camera_{i + 1}/info",
                type="sensor_msgs/msg/CameraInfo",
                serialization_format="cdr",
            )
        )

    return writer


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rgb-sources",
        help="folder with RGB or raw Bayer images",
        required=True,
        nargs="*",
    )
    parser.add_argument(
        "--intrinsic-params", help="yaml file with intrinsic parameters", required=True
    )
    parser.add_argument(
        "--detection-result", help="yaml with the result of detection", required=True
    )
    parser.add_argument(
        "--predictions",
        help="yaml with predicted trajectory points and covariance matrix",
    )
    parser.add_argument("--export", help="some_file.mcap", required=True)
    parser.add_argument(
        "--transform-cam-to-world",
        help="boolean flag whether to write and transform points "
        + "from to table coordinate frame",
        action="store_true",
    )

    return parser


def init_ball_marker(
    marker_id: int, current_time: int, position: List[int], camera_id: int, ttl=100
) -> Marker:
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

    msg.scale.x = 0.04
    msg.scale.y = 0.04
    msg.scale.z = 0.04

    msg.color.r = 1.0
    msg.color.g = 0.5
    msg.color.b = 0.0
    msg.color.a = 1.0

    msg.lifetime.nanosec = ttl * 10**6  # ttl in milliseconds

    return msg


def init_detection_center_marker(
    marker_id: int, current_time: int, position: List[int], camera_id: int, ttl=100
) -> ImageMarker:
    msg = ImageMarker()
    msg.header.frame_id = f"camera_{camera_id}"
    msg.header.stamp.sec = current_time // SEC_MULTIPLIER
    msg.header.stamp.nanosec = current_time % SEC_MULTIPLIER
    msg.ns = "ball_markers"
    msg.id = marker_id
    msg.type = ImageMarker.CIRCLE
    msg.action = ImageMarker.ADD

    msg.position.x = float(position[0])
    msg.position.y = float(position[1])
    msg.position.z = 0.0

    msg.scale = 3.0
    msg.filled = 255

    msg.fill_color.r = 1.0  # orange color
    msg.fill_color.g = 0.0
    msg.fill_color.b = 0.0
    msg.fill_color.a = 1.0  # alpha (1.0 = opaque, 0.0 = transparent)

    msg.outline_color.r = 1.0  # orange color
    msg.outline_color.g = 0.0
    msg.outline_color.b = 0.0
    msg.outline_color.a = 1.0  # alpha (1.0 = opaque, 0.0 = transparent)

    msg.lifetime.nanosec = ttl * 10**6  # ttl = 10ms

    return msg


def simulate(
    writer: rosbag2_py.SequentialWriter,
    rgb_sources,
    filename_to_info,
    trajectory_predictions,
    intrinsics,
    R: np.ndarray,
    T: np.ndarray,
) -> None:
    filenames_to_publish = sorted(filename_to_info.keys())
    cv_bridge_instance = CvBridge()

    current_simulation_time = 0  # in nanoseconds
    for i in range(len(filenames_to_publish)):
        if i % 10 == 0:
            publish_table_plain(writer, current_simulation_time)
        filename = filenames_to_publish[i]
        for camera_idx in range(2):
            # load and segmentate image
            image = cv2.imread(os.path.join(rgb_sources[camera_idx], filename))

            image = intrinsics[camera_idx].undistort(image)

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

            center_detection = init_detection_center_marker(
                get_new_marker_id(),
                current_simulation_time,
                filename_to_info[filename]["image_points"][camera_idx],
                camera_idx + 1,
                ttl=50,
            )
            writer.write(
                f"/camera_{camera_idx + 1}/ball_center",
                serialize_message(center_detection),
                current_simulation_time,
            )

        current_point = np.array(
            filename_to_info[filename]["triangulated_point"], dtype=float
        ).reshape((3, 1))
        ball_marker = init_ball_marker(
            get_new_marker_id(),
            current_simulation_time,
            (R @ current_point - T).flatten().tolist(),
            current_point.flatten().tolist(),
            ttl=FPS_LATENCY_MS,
        )
        ball_marker.header.frame_id = "table"
        writer.write(
            "/triangulation/ball_marker",
            serialize_message(ball_marker),
            current_simulation_time,
        )

        projection_marker = ball_marker
        projection_marker.id = get_new_marker_id()
        projection_marker.scale.z = 0.001
        projection_marker.pose.position.z = 0.0
        projection_marker.color.r = 1.0
        writer.write(
            "/triangulation/ball_table_projection",
            serialize_message(projection_marker),
            current_simulation_time,
        )

        if trajectory_predictions:
            publish_predicted_trajectory(
                writer, trajectory_predictions, filename, current_simulation_time
            )
        current_simulation_time += (
            FPS_LATENCY_MS * MS_MULTIPLIER
        )  # 15 ms between the frames


def publish_predicted_trajectory(
    writer: rosbag2_py.SequentialWriter,
    trajectory_dict: Dict[str, Dict[str, List]],
    filename: str,
    current_simulation_time: int,
) -> None:
    if filename not in trajectory_dict.keys():
        return
    # print("publishing trajectory")
    trajectory_data = trajectory_dict[filename]
    if len(trajectory_data["pred"]) != len(trajectory_data["var"]):
        print(
            "number of trajectory points and covarience matrices does not match: "
            + "{len(trajectory_data['pred'])} != {len(trajectory_data['var'])}"
        )

    prev_point = None
    for i in range(len(trajectory_data["pred"])):
        current_point = np.array(trajectory_data["pred"][i], dtype=float)
        eigenvalues, _ = np.linalg.eigh(
            np.array(trajectory_data["var"][i], dtype=float)
        )
        eigenvalues = np.sqrt(eigenvalues)

        msg = Marker()
        msg.header.frame_id = "table"
        msg.id = get_new_marker_id()
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ) = current_point.tolist()

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        msg.scale.x, msg.scale.y, msg.scale.z = eigenvalues.tolist()

        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 1.0
        msg.color.a = 0.6

        msg.lifetime.nanosec = FPS_LATENCY_MS * 10**6
        writer.write(
            "/triangulation/trajectory",
            serialize_message(msg),
            current_simulation_time,
        )

        # if both z coordintes are above the table
        if prev_point is None or current_point[2] * prev_point[2] > 0:
            prev_point = current_point
            continue

        # notation based on https://math.stackexchange.com/a/3584405
        q_1, q_2 = np.array([1, 0, 0], dtype=float), np.array([0, 1, 0], dtype=float)
        direction_vector = current_point - prev_point
        A = np.column_stack([q_1, q_2, -direction_vector])
        b = prev_point - q_1
        coef = np.linalg.solve(A, b)[2]
        intersection_point = prev_point + coef * direction_vector

        intersect_marker = Marker()
        intersect_marker.header.frame_id = "table"
        intersect_marker.id = get_new_marker_id()
        intersect_marker.type = Marker.SPHERE
        intersect_marker.action = Marker.ADD
        (
            intersect_marker.pose.position.x,
            intersect_marker.pose.position.y,
            intersect_marker.pose.position.z,
        ) = intersection_point.tolist()

        intersect_marker.pose.orientation.x = 0.0
        intersect_marker.pose.orientation.y = 0.0
        intersect_marker.pose.orientation.z = 0.0
        intersect_marker.pose.orientation.w = 1.0

        intersect_marker.scale.x, intersect_marker.scale.y, intersect_marker.scale.z = (
            0.04,
            0.04,
            0.001,
        )

        intersect_marker.color.r = 1.0
        intersect_marker.color.g = 0.0
        intersect_marker.color.b = 0.0
        intersect_marker.color.a = 1.0

        intersect_marker.lifetime.nanosec = FPS_LATENCY_MS * 10**6
        writer.write(
            "/triangulation/intersection_points",
            serialize_message(intersect_marker),
            current_simulation_time,
        )

        prev_point = current_point


def init_camera_info(
    writer: rosbag2_py.SequentialWriter, params_path: str, camera_ids=[1, 2]
) -> Tuple[List[CameraParameters], np.ndarray]:
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
    complanar_aruco_points = np.array(
        data["triangulated_common_points"], dtype=np.float64
    )

    centroid = np.mean(complanar_aruco_points, axis=0)
    _, _, VT = np.linalg.svd(
        complanar_aruco_points[:10] - centroid, full_matrices=False
    )
    print("normal is", VT[-1, :])
    return intrinsics, VT[-1, :]


def publish_table_plain(
    writer: rosbag2_py.SequentialWriter, simulation_time: int
) -> None:
    global last_table_marker_id
    if last_table_marker_id:
        marker = Marker()
        marker.id = last_table_marker_id
        marker.action = 2  # DELETE
        writer.write(
            "/triangulation/table_plane", serialize_message(marker), simulation_time
        )

    # publish blue table plain
    marker = Marker()
    marker.header.frame_id = "table"
    marker.id = get_new_marker_id()
    last_table_marker_id = marker.id
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = TABLE_LENGTH
    marker.scale.y = TABLE_WIDTH
    marker.scale.z = 0.01

    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.8
    marker.color.a = 0.3

    marker.pose.position.x = 0.0
    marker.pose.position.y = TABLE_LENGTH / 4
    marker.pose.position.z = 0.0
    (
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z,
        marker.pose.orientation.w,
    ) = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat().tolist()
    writer.write(
        "/triangulation/table_plane", serialize_message(marker), simulation_time
    )

    # publish white border
    marker = Marker()
    marker.id = get_new_marker_id()
    marker.header.frame_id = "table"
    marker.type = marker.LINE_STRIP
    marker.action = marker.ADD
    marker.scale.x = 0.01

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.9

    (
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z,
        marker.pose.orientation.w,
    ) = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat().tolist()

    coords = [
        (-TABLE_WIDTH / 2, -TABLE_LENGTH / 4),
        (-TABLE_WIDTH / 2, TABLE_LENGTH * 3 / 4),
        (TABLE_WIDTH / 2, TABLE_LENGTH * 3 / 4),
        (TABLE_WIDTH / 2, -TABLE_LENGTH / 4),
        (-TABLE_WIDTH / 2, -TABLE_LENGTH / 4),
    ]

    for cur_y, cur_x in coords:
        new_point = Point()
        new_point.x = cur_x
        new_point.y = cur_y
        marker.points.append(new_point)

    writer.write(
        "/triangulation/table_plane", serialize_message(marker), simulation_time
    )

    # publish length line
    marker.id = get_new_marker_id()
    marker.points = []

    coords = [
        (0.0, -TABLE_LENGTH / 4),
        (0.0, TABLE_LENGTH * 3 / 4),
    ]

    for cur_y, cur_x in coords:
        new_point = Point()
        new_point.x = cur_x
        new_point.y = cur_y
        marker.points.append(new_point)

    writer.write(
        "/triangulation/table_plane", serialize_message(marker), simulation_time
    )

    # publish width line
    marker.id = get_new_marker_id()
    marker.points = []

    coords = [
        (-TABLE_WIDTH / 2, TABLE_LENGTH / 4),
        (TABLE_WIDTH / 2, TABLE_LENGTH / 4),
    ]

    for cur_y, cur_x in coords:
        new_point = Point()
        new_point.x = cur_x
        new_point.y = cur_y
        marker.points.append(new_point)

    writer.write(
        "/triangulation/table_plane", serialize_message(marker), simulation_time
    )


def get_cam2world_transform(
    table_plane_normal: np.ndarray, table_orientation_points: List[List[float]]
) -> Tuple[np.ndarray]:
    edge_table_orient_point = np.array(table_orientation_points[0], dtype=float)
    middle_table_orient_point = np.array(table_orientation_points[1], dtype=float)
    x_vector = middle_table_orient_point - edge_table_orient_point
    z_vector = (
        table_plane_normal
        - (table_plane_normal.dot(x_vector) / np.linalg.norm(x_vector, ord=2))
        * x_vector
    )
    # z_vector = table_plane_normal
    y_vector = np.cross(x_vector, z_vector)

    x_vector /= np.linalg.norm(x_vector, ord=2)
    y_vector /= np.linalg.norm(y_vector, ord=2)
    z_vector /= np.linalg.norm(z_vector, ord=2)

    R = np.concatenate(
        (y_vector.reshape((3, 1)), x_vector.reshape((3, 1)), z_vector.reshape((3, 1))),
        axis=1,
    )
    table_line_middle = (edge_table_orient_point + middle_table_orient_point) / 2
    T = table_line_middle.reshape((3, 1))
    R_inv = np.linalg.inv(R)

    return R, T, R_inv, R_inv @ T, table_line_middle


def publish_cam2table_transform(
    writer: rosbag2_py.SequentialWriter, R: np.ndarray, T: np.ndarray
) -> None:
    static_transformation = TransformStamped()
    static_transformation.child_frame_id = "table"
    static_transformation.header.frame_id = "world"

    static_transformation.transform.translation.x = T[0, 0]
    static_transformation.transform.translation.y = T[1, 0]
    static_transformation.transform.translation.z = T[2, 0]
    static_transformation.header.stamp.sec = 0
    static_transformation.header.stamp.nanosec = 0

    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat().tolist()

    static_transformation.transform.rotation.x = qx
    static_transformation.transform.rotation.y = qy
    static_transformation.transform.rotation.z = qz
    static_transformation.transform.rotation.w = qw

    writer.write("/tf", serialize_message(static_transformation), 0)


if __name__ == "__main__":
    # init all modules
    parser = init_parser()
    args = parser.parse_args()
    writer = init_writer(args.export)

    # read and publish camera info
    intrinsics, table_plane_normal = init_camera_info(
        writer, args.intrinsic_params, [1, 2]
    )

    with open(args.detection_result, mode="r", encoding="utf-8") as file:
        data = json.load(file)

    trajectory_predictions = None
    if args.predictions:
        with open(args.predictions, mode="r", encoding="utf-8") as file:
            trajectory_predictions = json.load(file)

    R, T, R2table, T2table, table_center = get_cam2world_transform(
        table_plane_normal, data["table_orientation_points"]
    )
    publish_table_plain(writer, 0)
    publish_cam2table_transform(writer, R, T)

    simulate(
        writer,
        args.rgb_sources,
        data["triangulated_points"],
        trajectory_predictions,
        intrinsics,
        R2table,
        T2table,
    )
    del writer

    if not args.transform_cam_to_world:
        quit()

    for filename in data["triangulated_points"]:
        point = np.array(
            data["triangulated_points"][filename]["triangulated_point"], dtype=float
        ).reshape((3, 1))
        table_point = R2table @ point - T2table
        data["triangulated_points"][filename][
            "triangulated_point"
        ] = table_point.flatten().tolist()

    new_detection_filepath = args.detection_result[:-5] + "_table.json"
    with open(new_detection_filepath, mode="w", encoding="utf-8") as file:
        json.dump(data, file)
