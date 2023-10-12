#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <exception>

namespace handy {
struct CameraIntrinsicParameters {
    CameraIntrinsicParameters() = default;
    void save(const std::string path_to_yaml_file) const;
    int load(const std::string path_to_yaml_file);

    cv::Mat camera_matrix;
    cv::Vec<float, 5> dist_coefs;
};

struct CameraUndistortModule : CameraIntrinsicParameters {
    void initUndistortMaps(cv::Size& frame_size);
    cv::Mat& undistortImage(cv::Mat& src);

    std::pair<cv::Mat, cv::Mat> undistort_maps;

    cv::Mat undistortedImage;
};
}  // namespace handy
