#include "params.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <exception>

namespace handy {

CameraIntrinsicParameters::CameraIntrinsicParameters(
    cv::Size size, cv::Mat camera_matr, cv::Vec<float, 5> distort_coefs)
    : image_size(size), camera_matrix(camera_matr), dist_coefs(distort_coefs) {}

void CameraIntrinsicParameters::storeYaml(const std::string& yaml_path) const {
    std::ofstream param_file(yaml_path);
    if (!param_file) {
        throw std::invalid_argument("unable to open file");
    }

    YAML::Emitter output_yaml;
    output_yaml << YAML::BeginMap;  // global yaml map

    output_yaml << YAML::Key << "image_size";
    output_yaml << YAML::Value << YAML::BeginSeq << image_size.width << image_size.height
                << YAML::EndSeq;

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

    output_yaml << YAML::EndMap;  // global yaml map

    param_file << output_yaml.c_str();
    param_file.close();
}

CameraIntrinsicParameters CameraIntrinsicParameters::loadFromYaml(const std::string& yaml_path) {
    CameraIntrinsicParameters result{};

    const YAML::Node file = YAML::LoadFile(yaml_path);

    const std::vector<int> yaml_image_size = file["image_size"].as<std::vector<int>>();
    result.image_size = cv::Size(yaml_image_size[0], yaml_image_size[1]);

    const std::vector<double> yaml_camera_matrix = file["camera_matrix"].as<std::vector<double>>();
    result.camera_matrix = cv::Mat(yaml_camera_matrix, true);

    const std::vector<float> coefs = file["distorsion_coefs"].as<std::vector<float>>();
    result.dist_coefs = cv::Mat(coefs, true);

    result.initUndistortMaps();

    return result;
}

void CameraIntrinsicParameters::initUndistortMaps() {
    // note that new camera matrix equals initial camera matrix
    // because neither scaling nor cropping is used when undistoring
    cv::initUndistortRectifyMap(
        camera_matrix,
        dist_coefs,
        cv::noArray(),
        camera_matrix,  // newCameraMatrix == this->camera_matrix
        image_size,
        CV_16SC2,
        cached.undistort_maps.first,
        cached.undistort_maps.second);
    cached.undistortedImage = cv::Mat(image_size, CV_8UC3);
}

cv::Mat CameraIntrinsicParameters::undistortImage(cv::Mat& src) {
    cv::remap(
        src,
        cached.undistortedImage,
        cached.undistort_maps.first,
        cached.undistort_maps.second,
        cv::INTER_NEAREST);
    return cached.undistortedImage;
}

}  // namespace handy
