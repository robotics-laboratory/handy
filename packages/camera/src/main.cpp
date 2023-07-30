#include <rclcpp/rclcpp.hpp>

#include "camera.h"

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<handy::camera::CameraNode>());
    rclcpp::shutdown();

    return 0;
}
