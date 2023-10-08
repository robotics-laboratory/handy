#include "params.h"

namespace handy {

void CameraIntrinsicParameters::save(const std::string path_to_yaml_file) const {
    cv::FileStorage file(path_to_yaml_file, cv::FileStorage::WRITE);
    file << "camera_matrix" << camera_matrix << "distortion_coefs" << dist_coefs;
    file.release();
}

int CameraIntrinsicParameters::load(const std::string path_to_yaml_file) {
    const YAML::Node file = YAML::LoadFile(path_to_yaml_file);
    const std::vector<float> yaml_camera_matrix = file["camera_matrix"].as<std::vector<float>>();
    camera_matrix = cv::Mat(yaml_camera_matrix, true);
    for (int i = 0; i < 9; ++i) {
        camera_matrix.at<float>(i / 3, i % 3) = yaml_camera_matrix[i];
    }
    const std::vector<float> coefs = file["distorsion_coefs"].as<std::vector<float>>();
        for (int i = 0; i < 5; ++i) {
            dist_coefs[i] = coefs[i];
        }

    return 0;
}

void CameraUndistortModule::initUndistortMaps(cv::Size& frame_size) {
    // note that new camera matrix equals initial camera matrix
    // because neither scaling nor cropping is used when undistoring
    cv::initUndistortRectifyMap(
        camera_matrix,
        dist_coefs,
        cv::noArray(),
        camera_matrix,  // newCameraMatrix == this->camera_matrix
        frame_size,
        CV_16SC2,
        undistort_maps.first,
        undistort_maps.second);
    undistortedImage = cv::Mat(frame_size, CV_8UC3);
}
cv::Mat& CameraUndistortModule::undistortImage(cv::Mat& src) {
    cv::remap(
        src, undistortedImage, undistort_maps.first, undistort_maps.second, cv::INTER_NEAREST);
}

}  // namespace handy