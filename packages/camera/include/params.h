#pragma once

#include <opencv2/core/core.hpp>
#include <optional>

namespace handy {

struct CameraUndistortModule;

struct CameraIntrinsicParameters {
    CameraIntrinsicParameters() = default;
    CameraIntrinsicParameters(const std::string& path_to_yaml_file, const std::string& calib_name);
    void save() const;
    static CameraUndistortModule load(
        const std::string& path_to_yaml_file, const std::string& calib_name,
        std::optional<cv::Size> frame_size = std::nullopt);

    cv::Mat camera_matrix;
    cv::Vec<float, 5> dist_coefs;
    std::string path_to_yaml_file;
    std::string calib_name;
};

struct CameraUndistortModule : CameraIntrinsicParameters {
    CameraUndistortModule(const std::string& path_to_yaml_file, const std::string& calib_name);
    void initUndistortMaps(cv::Size& frame_size);
    void initUndistortMaps(std::optional<cv::Size> frame_size);
    cv::Mat undistortImage(cv::Mat& src);

    std::pair<cv::Mat, cv::Mat> undistort_maps;
    cv::Mat undistortedImage;
};
}  // namespace handy
