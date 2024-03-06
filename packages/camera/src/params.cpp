#include "params.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <exception>
#include <fstream>

namespace handy {

CameraIntrinsicParameters::CameraIntrinsicParameters(
    cv::Size size, cv::Mat camera_matr, const cv::Vec<double, 5>& distort_coefs, const int cam_id)
    : image_size(size)
    , camera_matrix(std::move(camera_matr))
    , dist_coefs(distort_coefs)
    , camera_id(cam_id) {}

void CameraIntrinsicParameters::storeYaml(const std::string& yaml_path) const {
    const std::string camera_id_str = std::to_string(camera_id);
    YAML::Node config;
    std::ifstream param_file(yaml_path);
    if (param_file) {
        config = YAML::Load(param_file);
        param_file.close();
    }

    YAML::Node&& intrinsics = config["intrinsics"];
    YAML::Node&& camera_id_node = intrinsics[camera_id_str];

    camera_id_node["image_size"] = YAML::Node(YAML::NodeType::Sequence);
    camera_id_node["image_size"].push_back(image_size.width);
    camera_id_node["image_size"].push_back(image_size.height);

    camera_id_node["camera_matrix"] = YAML::Node(YAML::NodeType::Sequence);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_id_node["camera_matrix"].push_back(camera_matrix.at<double>(i, j));
        }
    }

    camera_id_node["distortion_coefs"] = YAML::Node(YAML::NodeType::Sequence);
    for (int i = 0; i < 5; ++i) {
        camera_id_node["distortion_coefs"].push_back(dist_coefs[i]);
    }

    std::ofstream output_file(yaml_path);
    if (!output_file) {
        throw std::invalid_argument("unable to open file");
    }
    output_file << config;
    output_file.close();
}

CameraIntrinsicParameters CameraIntrinsicParameters::loadFromYaml(
    const std::string& yaml_path, const int camera_id) {
    CameraIntrinsicParameters result{};
    result.camera_id = camera_id;
    const std::string camera_id_str = std::to_string(camera_id);

    const YAML::Node intrinsics = YAML::LoadFile(yaml_path)["intrinsics"];

    const auto yaml_image_size = intrinsics[camera_id_str]["image_size"].as<std::vector<int>>();
    result.image_size = cv::Size(yaml_image_size[0], yaml_image_size[1]);

    const auto yaml_camera_matrix =
        intrinsics[camera_id_str]["camera_matrix"].as<std::vector<double>>();
    result.camera_matrix = cv::Mat(yaml_camera_matrix, true);

    const auto coefs = intrinsics[camera_id_str]["distorsion_coefs"].as<std::vector<float>>();
    result.dist_coefs = cv::Mat(coefs, true);

    result.initUndistortMaps();

    return result;
}

CameraIntrinsicParameters CameraIntrinsicParameters::loadFromParams(
    cv::Size param_image_size, const std::vector<double>& param_camera_matrix,
    const std::vector<double>& param_dist_coefs) {
    CameraIntrinsicParameters result{};

    result.image_size = param_image_size;
    result.camera_matrix = cv::Mat(param_camera_matrix, true);
    result.dist_coefs = cv::Mat(param_dist_coefs, true);
    result.initUndistortMaps();

    return result;
}

void CameraIntrinsicParameters::initUndistortMaps() {
    // note that new camera matrix equals initial camera matrix
    // because neither scaling nor cropping is used when undistoring
    cv::initUndistortRectifyMap(
        result.camera_matrix,
        result.dist_coefs,
        cv::noArray(),
        result.camera_matrix,  // newCameraMatrix == this->camera_matrix
        result.image_size,
        CV_16SC2,
        result.cached.undistort_maps.first,
        result.cached.undistort_maps.second);
    result.cached.undistortedImage = cv::Mat(result.image_size, CV_8UC3);

    return result;
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
