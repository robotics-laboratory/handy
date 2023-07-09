#include "camera_node.h"


CameraNode::CameraNode() : Node("camera_node") {
    camera_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera_image_topic", 100);
    test_pub = this->create_publisher<geometry_msgs::msg::Point>("test_topic", 100);
    camera_.open(0);
    if (!this->camera_.isOpened()) {
        RCLCPP_INFO(this->get_logger(), "ERROR, camera not opened!");
        return;
    }

    timer_ = this->create_wall_timer(200ms, std::bind(&CameraNode::cameraHandler, this));
    test_timer_ = this->create_wall_timer(100ms, std::bind(&CameraNode::pub_next_value, this));


}

void CameraNode::pub_next_value() {
    geometry_msgs::msg::Point msg;
    msg.x = std::cos(3.1415f * cnt / 360) * 100;
    msg.y = std::sin(3.1415f * cnt / 360) * 100;
    msg.z = cnt;

    test_pub->publish(msg);
    cnt = (cnt + 1) % 360;
}
void CameraNode::cameraHandler() {
    cv::Mat frame;
    this->camera_.read(frame);
    if (frame.empty()) {
        RCLCPP_INFO(this->get_logger(), "ERROR, frame not recieved!");
        return;
    }
    //cv::imwrite("test_image.png", frame);
    RCLCPP_INFO(this->get_logger(), "Got image with %d channels of width %d and height %d ", frame.channels(), frame.cols, frame.rows);
    publishImage(frame);
}

bool CameraNode::publishImage(const cv::Mat &image) {
    sensor_msgs::msg::Image img;
    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg(img);
    camera_image_pub_->publish(img);
    RCLCPP_INFO(this->get_logger(), "Sent image with %d channels of width %d and height %d ", image.channels(), image.cols, image.rows);
    return true;
}

std_msgs::msg::Header CameraNode::constructHeader() {
    std_msgs::msg::Header header;
    header.stamp = now();
    return header;
}



int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();

    return 0;
}