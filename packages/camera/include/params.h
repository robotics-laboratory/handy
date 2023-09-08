#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <tuple>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

namespace handy {
struct CameraIntrinsicParameters {
  public:
    void save(const std::string path_to_yaml_file) const;
    int load(const std::string path_to_yaml_file, rclcpp::Logger logger);

    cv::Mat camera_matrix = cv::Mat(3, 3, CV_16FC1);
    cv::Vec<float, 5> dist_coefs;
    cv::Mat new_camera_matrix;
    std::pair<cv::Mat, cv::Mat> undistort_maps;
};
}  // namespace handy
