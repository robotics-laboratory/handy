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
    void save(const std::string path_to_yaml_file) const;
    int load(const std::string path_to_yaml_file, rclcpp::Logger logger);

    cv::Mat camera_matrix;
    cv::Vec<float, 5> dist_coefs;
    std::vector<cv::Mat> rotation_vectors;
    std::vector<cv::Mat> translation_vectors;
    cv::Mat new_camera_matrix;
    std::pair<cv::Mat, cv::Mat> undistort_maps;
};
}  // namespace handy
