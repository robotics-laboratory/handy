from rclpy.serialization import serialize_message
from visualization_msgs.msg import Marker
import rosbag2_py
import argparse
from builtin_interfaces.msg import Duration, Time

writer = rosbag2_py.SequentialWriter()
writer.open(
    rosbag2_py.StorageOptions(uri="output.mcap", storage_id="mcap"),
    rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    ),
)

writer.create_topic(
    rosbag2_py.TopicMetadata(
        name="/ball_marker", type="visualization_msgs/msg/Marker", serialization_format="cdr"
    )
)

n = int(input())
new_line = input()
for i in range(n):
    current_time = 10 * i * 10**6
    msg = Marker()
    msg.header.frame_id = "camera_1"
    msg.header.stamp.sec = current_time // 10**9
    msg.header.stamp.nanosec = current_time % 10**9
    msg.ns = "ball_markers"
    msg.id = i
    msg.type = Marker.SPHERE
    msg.action = Marker.ADD
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, new_line.split())
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
    msg.lifetime.nanosec = 10 * 10**6 # ttl = 10ms

    writer.write("/ball_marker", serialize_message(msg), 10 * i * 10**6)
    try:
        new_line = input()
    except EOFError:
        break

del writer

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask-sources", help="folder with masks", required=True, nargs="*")
    parser.add_argument("--rgb-sources", help="folder with masks", required=True, nargs="*")
    parser.add_argument("--intrinsic-params", help="yaml file with intrinsic parameters", required=True)
    parser.add_argument("--detection-result", help="yaml with the result of detection", required=True)
    parser.add_argument("--export", help="some_file.mcap", required=True)

    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    