#include "params.h"

namespace handy {

void CameraIntrinsicParameters::save(const std::string path_to_yaml_file) const {
    cv::FileStorage file(path_to_yaml_file, cv::FileStorage::WRITE);
    file << "camera_matrix" << camera_matrix << "distortion_coefs" << dist_coefs;
    file.release();
}

int CameraIntrinsicParameters::load(const std::string path_to_yaml_file, rclcpp::Logger logger) {
    try {
        cv::FileStorage file(path_to_yaml_file, cv::FileStorage::READ);
        if (file["camera_matrix"].isNone() || file["distortion_coefs"].isNone()) {
            return -1;
        }
        RCLCPP_INFO_STREAM(logger, "read and checked");
        cv::Mat test;
        file["camera_matrix"] >> test;
        std::memcpy(&camera_matrix.data[0], &test.data[0], 9);
        RCLCPP_INFO_STREAM(logger, camera_matrix.at<float>(2, 2));
        //file["camera_matrix"] >> camera_matrix;
        RCLCPP_INFO_STREAM(logger, "wrote to cam matrix");
        RCLCPP_INFO_STREAM(logger, camera_matrix.data);

        // file["distortion_coefs"] >> dist_coefs;
        file.release();
        return 0;

    } catch (const std::exception& e) {
        RCLCPP_INFO_STREAM(logger, e.what());
        return -1;
    }
}
}  // namespace handy