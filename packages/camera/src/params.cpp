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
        cv::Mat cam_matrix;
        file["camera_matrix"] >> cam_matrix;
        RCLCPP_INFO_STREAM(logger, "initing");
        camera_matrix = cv::Mat(3, 3, CV_32F, cam_matrix.data);
        
        //camera_matrix = matCopy(cam_matrix);
        //this->camera_matrix = cam_matrix.clone();
        //cam_matrix.copyTo(this->camera_matrix);
        //file["camera_matrix"] >> camera_matrix;

        file["distortion_coefs"] >> dist_coefs;
        file.release();
        return 0;

    } catch (const std::exception& e) {
        RCLCPP_INFO_STREAM(logger, e.what());
        return -1;
    }
}
}  // namespace handy