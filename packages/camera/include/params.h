#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include <tuple>

namespace handy {
struct CameraIntrinsicParameters {
    CameraIntrinsicParameters() = default;

    CameraIntrinsicParameters(
        cv::Size size, cv::Mat camera_matrix, const cv::Vec<double, 5>& distort_coefs,
        int cam_id = 0);

    cv::Mat undistortImage(cv::Mat& src);

    void storeYaml(const std::string& yaml_path) const;
    static CameraIntrinsicParameters loadFromYaml(const std::string& yaml_path, int camera_id = 0);
    static bool saveStereoCalibration(
        const std::string& yaml_path, cv::Mat& rotation_vectors, cv::Mat& translation_vectors,
        cv::Size& image_size);
    static void loadStereoCalibration(
        const std::string& yaml_path, cv::Mat& rotation_vectors, cv::Mat& translation_vectors,
        cv::Size& image_size);

    int camera_id = 0;
    cv::Size image_size{};
    cv::Mat camera_matrix{};
    cv::Vec<double, 5> dist_coefs{};

    struct Cached {
        std::pair<cv::Mat, cv::Mat> undistort_maps{};
        cv::Mat undistortedImage{};
    } cached{};
};
}  // namespace handy
