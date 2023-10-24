#pragma once

#include <opencv2/core/core.hpp>
#include <optional>

namespace handy {

struct CameraUndistortModule;

struct CameraIntrinsicParameters {
    CameraIntrinsicParameters() = default;
    void save(const std::string& path_to_yaml_file) const;
    static CameraUndistortModule load(
        const std::string& path_to_yaml_file, std::optional<cv::Size> frame_size = std::nullopt);

    cv::Mat camera_matrix;
    cv::Vec<float, 5> dist_coefs;
};

struct CameraUndistortModule : CameraIntrinsicParameters {
    void initUndistortMaps(cv::Size& frame_size);
    void initUndistortMaps(std::optional<cv::Size> frame_size);
    cv::Mat undistortImage(cv::Mat& src);

    std::pair<cv::Mat, cv::Mat> undistort_maps;
    cv::Mat undistortedImage;
};
}  // namespace handy
