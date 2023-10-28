#include "params.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <exception>

namespace handy {

CameraIntrinsicParameters::CameraIntrinsicParameters(
    const std::string& path_to_yaml, const std::string& name)
    : path_to_yaml_file(path_to_yaml), calib_name(name) {}

void CameraIntrinsicParameters::save() const {
    std::ofstream param_file(path_to_yaml_file);
    if (!param_file) {
        throw std::invalid_argument("unable to open file");
    }

    YAML::Emitter output_yaml;
    output_yaml << YAML::BeginMap; // global yaml map

    output_yaml << YAML::Key << calib_name << YAML::BeginMap; // calib_name map
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

    output_yaml << YAML::EndMap; // calib_name map
    output_yaml << YAML::EndMap; // global yaml map

    param_file << output_yaml.c_str();
    param_file.close();
}

CameraUndistortModule CameraIntrinsicParameters::load(
    const std::string& path_to_yaml_file, const std::string& calib_name,
    std::optional<cv::Size> frame_size) {
    CameraUndistortModule result(path_to_yaml_file, calib_name);
    
    const YAML::Node file = YAML::LoadFile(path_to_yaml_file);
    const std::vector<double> yaml_camera_matrix = file[calib_name]["camera_matrix"].as<std::vector<double>>();
    result.camera_matrix = cv::Mat(yaml_camera_matrix, true);

    const std::vector<float> coefs = file[calib_name]["distorsion_coefs"].as<std::vector<float>>();
    result.dist_coefs = cv::Mat(coefs, true);

    if (frame_size) {
        result.initUndistortMaps(frame_size);
    }
    return result;
}

CameraUndistortModule::CameraUndistortModule(
    const std::string& path_to_yaml, const std::string& name) {
    path_to_yaml_file = path_to_yaml;
    calib_name = name;
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

void CameraUndistortModule::initUndistortMaps(std::optional<cv::Size> frame_size) {
    initUndistortMaps(*frame_size);
}

cv::Mat CameraUndistortModule::undistortImage(cv::Mat& src) {
    cv::remap(
        src, undistortedImage, undistort_maps.first, undistort_maps.second, cv::INTER_NEAREST);
    return undistortedImage;
}

}  // namespace handy