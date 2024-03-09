#include "calibration.h"

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);

    rclcpp::Node::SharedPtr node = std::make_shared<handy::calibration::CalibrationNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();

    return 0;
}
