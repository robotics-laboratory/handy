#include "calibration.h"

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<handy::calibration::CalibrationNode>());
    rclcpp::shutdown();

    return 0;
}