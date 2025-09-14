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

from scipy.ndimage import center_of_mass
from typing import Tuple, List

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
        yaw_pitch_roll_order=False,
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
        if yaw_pitch_roll_order:
            self.rotation_matrix = Rotation.from_euler(
                "xyz", rotation_vector, degrees=True
            ).as_matrix()
        else:
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

    def undistort(self, image: cv2.Mat) -> cv2.Mat:
        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_NEAREST)

    def publish_camera_info(self, writer: rosbag2_py.SequentialWriter, current_time: int) -> None:
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

    def publish_transform(self, writer: rosbag2_py.SequentialWriter, current_time: int) -> None:
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
        "--params-file",
        help="yaml file with intrinsic parameters and cameras' positions",
        required=True,
    )
    parser.add_argument("--export", help="some_file.mcap", required=True)

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
    complanar_aruco_points = np.array(data["triangulated_common_points"], dtype=np.float64)

    centroid = np.mean(complanar_aruco_points, axis=0)
    _, _, VT = np.linalg.svd(complanar_aruco_points[:10] - centroid, full_matrices=False)
    print("normal is", VT[-1, :])
    return intrinsics, VT[-1, :]


def publish_table_plain(writer: rosbag2_py.SequentialWriter, simulation_time: int) -> None:
    global last_table_marker_id
    if last_table_marker_id:
        marker = Marker()
        marker.id = last_table_marker_id
        marker.action = 2  # DELETE
        writer.write("/triangulation/table_plane", serialize_message(marker), simulation_time)

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
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    (
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z,
        marker.pose.orientation.w,
    ) = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat().tolist()
    writer.write("/triangulation/table_plane", serialize_message(marker), simulation_time)

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
        (-TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
        (-TABLE_WIDTH / 2, TABLE_LENGTH / 2),
        (TABLE_WIDTH / 2, TABLE_LENGTH / 2),
        (TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
        (-TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
    ]

    for cur_y, cur_x in coords:
        new_point = Point()
        new_point.x = cur_x
        new_point.y = cur_y
        marker.points.append(new_point)

    writer.write("/triangulation/table_plane", serialize_message(marker), simulation_time)

    # publish length line
    marker.id = get_new_marker_id()
    marker.points = []

    coords = [
        (0.0, -TABLE_LENGTH / 2),
        (0.0, TABLE_LENGTH / 2),
    ]

    for cur_y, cur_x in coords:
        new_point = Point()
        new_point.x = cur_x
        new_point.y = cur_y
        marker.points.append(new_point)

    writer.write("/triangulation/table_plane", serialize_message(marker), simulation_time)

    # publish width line
    marker.id = get_new_marker_id()
    marker.points = []

    coords = [
        (-TABLE_WIDTH / 2, 0.0),
        (TABLE_WIDTH / 2, 0.0),
    ]

    for cur_y, cur_x in coords:
        new_point = Point()
        new_point.x = cur_x
        new_point.y = cur_y
        marker.points.append(new_point)

    writer.write("/triangulation/table_plane", serialize_message(marker), simulation_time)


def get_cam2world_transform(
    table_plane_normal: np.ndarray, table_orientation_points: List[List[float]]
) -> Tuple[np.ndarray]:
    edge_table_orient_point = np.array(table_orientation_points[0], dtype=float)
    middle_table_orient_point = np.array(table_orientation_points[1], dtype=float)
    x_vector = middle_table_orient_point - edge_table_orient_point
    z_vector = (
        table_plane_normal
        - (table_plane_normal.dot(x_vector) / np.linalg.norm(x_vector, ord=2)) * x_vector
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


from typing import Any


class Transformation:
    def __init__(self, R, t):
        self.R = R
        self.t = t.reshape((3, 1))
        self.R_inv = np.linalg.inv(self.R)

    def __call__(self, point):
        return self.R @ point + self.t

    def __mul__(self, other):
        return Transformation(self.R @ other.R, self.R @ other.t + self.t)

    def transform(self, point):
        return self(point)

    def inverse_transform(self, point):
        return self.R_inv @ (point - self.t)

    # right transformation is applied first
    def __mult__(self, other):
        return Transformation(self.R @ other.R, self.t + other.t)


class Image:
    def __init__(
        self, camera_matrix, camera_transformation, distortion_coefs, image_size=(1200, 1920)
    ):
        self.camera_matrix = camera_matrix
        self.camera_transformation = camera_transformation
        self.distortion_coefs = distortion_coefs
        self.image_size = image_size

    def normilise_image_point(self, point):
        x_normalised = (point[0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_normalised = (point[1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        return np.array([x_normalised, y_normalised, 1]).reshape(3, 1)

    # in world coordinates
    def project_point_to_image(self, point):
        if point.shape != (3, 1):
            point = point.reshape((3, 1))
        return self.camera_matrix @ self.camera_transformation(point)

        # transformed_point = self.camera_transformation(point)
        # # transformed_point = transformed_point / transformed_point[2]
        # projected_point = self.camera_matrix @ transformed_point
        # return projected_point

    def project_points_to_image(self, points):
        return np.array([self.project_point_to_image(point) for point in points])

    def normilize_image_point(self, image_point):
        x_normalised = (image_point[0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_normalised = (image_point[1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        return np.array([x_normalised, y_normalised, 1]).reshape(3, 1)

    def project_ball_to_image(self, center, radius: float) -> np.ndarray:
        def valid_coords(x, y):
            return x >= 0 and x < self.image_size[1] and y >= 0 and y < self.image_size[0]

        center = center.reshape((3, 1))
        camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        transformed_center = self.camera_transformation(center)
        projected_center = self.camera_matrix @ transformed_center
        projected_center /= projected_center[2]

        if (
            np.linalg.norm(
                projected_center.flatten()
                - np.array([self.image_size[1] / 2, self.image_size[0] / 2, 1])
            )
            > 2000
        ):
            return np.zeros(self.image_size)

        image = np.zeros(self.image_size)
        checked_pixels = set()

        pixels_to_check = {(int(projected_center[0][0]), int(projected_center[1][0]))}
        while pixels_to_check:
            x, y = pixels_to_check.pop()

            image_point_camera_ray = camera_matrix_inv @ np.array([x, y, 1]).reshape((3, 1))
            image_point_world_ray = self.camera_transformation.inverse_transform(
                image_point_camera_ray
            ) - self.camera_transformation.inverse_transform(np.array([0, 0, 0]).reshape((3, 1)))
            ball_center_world_ray = center - self.camera_transformation.inverse_transform(
                np.array([0, 0, 0]).reshape((3, 1))
            )

            distance = np.linalg.norm(
                np.cross(ball_center_world_ray.flatten(), image_point_world_ray.flatten()), ord=2
            ) / np.linalg.norm(image_point_world_ray, ord=2)
            if distance <= radius:
                if valid_coords(x, y):
                    image[y, x] = 1
                # adding all 8 neighbours to the queue
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (x + dx, y + dy) not in checked_pixels:
                            pixels_to_check.add((x + dx, y + dy))
                            checked_pixels.add((x + dx, y + dy))

        return image


def get_bbox(mask: np.ndarray) -> List[float]:
    if not np.any(mask):
        return [0.0, 0.0, 0.0, 0.0]
    # x_min, y_min, x_max, y_max
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    bbox = list(map(float, [x1, y1, x2, y2]))
    return bbox


def get_mask_center(mask):
    bbox = get_bbox(mask)
    centroid_x = (bbox[0] + bbox[2]) / 2
    centroid_y = (bbox[1] + bbox[3]) / 2
    return np.array([centroid_x, centroid_y])


def get_mask_centroid(mask):
    return np.array(center_of_mass(mask))


class StereoScene:
    def __init__(self, left_camera: Image, right_camera: Image, table_middle_normal):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.table_middle_normal = table_middle_normal
        self.last_n_clicks = 0

    def project_ball_to_images(self, center, radius):
        left_image = self.left_camera.project_ball_to_image(center, radius)
        right_image = self.right_camera.project_ball_to_image(center, radius)
        return left_image, right_image

    def triangulate_position(self, points_by_view_1, points_by_view_2, world2cam, cam2cam):
        print("triangulating")
        # print(points_by_view)
        world2cam_Rt = np.column_stack((world2cam.R, world2cam.t))
        world2second_cam = cam2cam * world2cam
        world2second_cam_Rt = np.column_stack((world2second_cam.R, world2second_cam.t))
        proj_1 = self.left_camera.camera_matrix @ world2cam_Rt
        proj_2 = self.right_camera.camera_matrix @ world2second_cam_Rt

        res = cv2.triangulatePoints(proj_1, proj_2, points_by_view_1, points_by_view_2)
        res /= res[3, :]  # normalizing

        # TODO preserve 4D points?
        return res[:3, :]


def evaluate_camera_position(
    world2master: Transformation,
    master2second: Transformation,
    center_extractor,
    camera_params_1: CameraParameters,
    camera_params_2: CameraParameters,
    simulation_time
):
    NUMBER_OF_SPHERES = 6
    image_1 = Image(camera_params_1.camera_matrix, world2master, camera_params_1.dist_coefs)
    image_2 = Image(
        camera_params_2.camera_matrix, master2second * world2master, camera_params_2.dist_coefs
    )
    stereo_scene = StereoScene(image_1, image_2, None)

    sphere_centers = []
    for y in np.linspace(-TABLE_LENGTH / 2, TABLE_LENGTH / 2, NUMBER_OF_SPHERES):
        for x in np.linspace(-TABLE_WIDTH / 2, TABLE_WIDTH / 2, NUMBER_OF_SPHERES):
            for z in np.linspace(0, 1, NUMBER_OF_SPHERES):
                sphere_centers.append((x, y, z))

    sphere_centers = np.array(sphere_centers).T
    points_1 = []
    points_2 = []
    valid_sphere_centers = []
    # world2second = master2second * world2master


    for i in range(sphere_centers.shape[1]):
        mask_1, mask_2 = stereo_scene.project_ball_to_images(sphere_centers[:, i : (i + 1)], 0.02)
        ball_marker = init_ball_marker(get_new_marker_id(), simulation_time, sphere_centers[:, i : (i + 1)].flatten(), 1, ttl=SEC_MULTIPLIER)
        ball_marker.header.frame_id = "table"
        writer.write(
            "/triangulation/ball_marker",
            serialize_message(ball_marker),
            simulation_time,
        )

        if np.sum(mask_1) == 0 or np.sum(mask_2) == 0:
            continue

        points_1.append(center_extractor(mask_1))
        points_2.append(center_extractor(mask_2))
        valid_sphere_centers.append(sphere_centers[:, i])

    points_1 = np.array(points_1).T
    points_2 = np.array(points_2).T
    sphere_centers = np.array(valid_sphere_centers).T

    triangulated_points = stereo_scene.triangulate_position(
        points_1, points_2, world2master, master2second
    )

    # Calculate the Euclidean distance between the true and triangulated points
    distances = np.linalg.norm(sphere_centers - triangulated_points, axis=0)

    # Calculate mean and standard deviation of the distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    print(f"Mean distance error: {mean_distance}")
    print(f"Standard deviation of distance error: {std_distance}")
    print("evalution complete")
    print()


if __name__ == "__main__":
    # init all modules
    parser = init_parser()
    args = parser.parse_args()
    writer = init_writer(args.export)

    table_plane_normal = np.array([0, 0, 1])
    publish_table_plain(writer, 100)

    with open(args.params_file, mode="r", encoding="utf-8") as yaml_params_file:
        params = yaml.safe_load(yaml_params_file)
    yaml_intrinsics_dics = params["intrinsics"]
    publish_cam2table_transform(writer, np.eye(3), np.zeros((3, 1)))

    for position_idx in params["camera_positions"].keys():
        print(position_idx)
        R_1, T_1 = (
            params["camera_positions"][position_idx][1]["rotation"],
            params["camera_positions"][position_idx][1]["translation"],
        )
        R_1 = np.array(R_1)
        T_1 = np.array(T_1)

        R_2, T_2 = (
            params["camera_positions"][position_idx][2]["rotation"],
            params["camera_positions"][position_idx][2]["translation"],
        )
        R_2 = np.array(R_2)
        T_2 = np.array(T_2)

        intrinsics = [
            CameraParameters(
                yaml_intrinsics_dics["image_size"],
                yaml_intrinsics_dics[1]["camera_matrix"],
                yaml_intrinsics_dics[1]["distortion_coefs"],
                1,
                R_1,
                T_1,
                yaw_pitch_roll_order=True,
            ),
            CameraParameters(
                yaml_intrinsics_dics["image_size"],
                yaml_intrinsics_dics[2]["camera_matrix"],
                yaml_intrinsics_dics[2]["distortion_coefs"],
                2,
                R_2,
                T_2,
                yaw_pitch_roll_order=True,
            ),
        ]
        for cam_params in intrinsics:
            cam_params.publish_transform(writer, SEC_MULTIPLIER * position_idx)
            cam_params.publish_camera_info(writer, SEC_MULTIPLIER * position_idx)
            publish_table_plain(writer, SEC_MULTIPLIER * position_idx)

            world2master = Transformation(
                intrinsics[0].rotation_matrix, intrinsics[1].translation_vector
            )
            rotation_master2slave = world2master.R_inv @ world2master.R
            translation_master2slave = intrinsics[1].translation_vector
            # print(rotation_master2slave.shape)
            master2slave = Transformation(rotation_master2slave, translation_master2slave)

            evaluate_camera_position(
                world2master,
                master2slave,
                get_mask_center,
                intrinsics[0],
                intrinsics[1],
                SEC_MULTIPLIER * position_idx
            )


    del writer
