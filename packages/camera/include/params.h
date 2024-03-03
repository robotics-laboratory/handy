#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include <tuple>

namespace handy {
struct CameraIntrinsicParameters {
    CameraIntrinsicParameters() = default;

    CameraIntrinsicParameters(
        cv::Size size, cv::Mat camera_matrix, const cv::Vec<float, 5>& distort_coefs);

    void initUndistortMaps();
    cv::Mat undistortImage(cv::Mat& src);

    void storeYaml(const std::string& yaml_path) const;
    static CameraIntrinsicParameters loadFromYaml(const std::string& yaml_path);

    cv::Size image_size{};
    cv::Mat camera_matrix{};
    cv::Vec<float, 5> dist_coefs{};

    struct Cached {
        std::pair<cv::Mat, cv::Mat> undistort_maps{};
        cv::Mat undistortedImage{};
    } cached{};
};
}  // namespace handy
