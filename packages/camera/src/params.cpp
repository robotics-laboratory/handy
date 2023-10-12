#include "params.h"

namespace handy {

void CameraIntrinsicParameters::save(const std::string path_to_yaml_file) const {
    std::ofstream param_file(path_to_yaml_file);
    if (!param_file) {
        throw std::invalid_argument("unable to open file");
    }

    YAML::Emitter output_yaml;
    output_yaml << YAML::BeginMap;

    output_yaml << YAML::Key << "camera_matrix";
    output_yaml << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            output_yaml << camera_matrix.at<double>(i, j);
        }
    }
    output_yaml << YAML::EndSeq;

    output_yaml << YAML::Key << "distorsion_coefs";
    output_yaml << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < 5; ++i) {
        output_yaml << dist_coefs[i];
    }
    output_yaml << YAML::EndSeq;

    output_yaml << YAML::EndMap;

    param_file << output_yaml.c_str();
    param_file.close();
}

int CameraIntrinsicParameters::load(const std::string path_to_yaml_file) {
    const YAML::Node file = YAML::LoadFile(path_to_yaml_file);
    const std::vector<double> yaml_camera_matrix = file["camera_matrix"].as<std::vector<double>>();
    camera_matrix = cv::Mat(yaml_camera_matrix, true);

    const std::vector<float> coefs = file["distorsion_coefs"].as<std::vector<float>>();
    dist_coefs = cv::Mat(coefs, true);

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
    return undistortedImage;
}

}  // namespace handy