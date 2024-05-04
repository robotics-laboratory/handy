#include "triangulation.h"

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/calib3d.hpp>
// #include <opencv2/sfm/projection.hpp>

namespace handy {
TriangulationNode::TriangulationNode(std::string& params_file_path, std::vector<int>& camera_ids) {
    if (camera_ids.size() != 2) {
        printf("only 2 cameras are supported\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < camera_ids.size(); ++i) {
        param_.cameras_intrinsics.push_back(
            CameraIntrinsicParameters::loadFromYaml(params_file_path, camera_ids[i]));
        param_.camera_stereo_params.emplace_back();
        cv::Mat rotation_vector;
        CameraIntrinsicParameters::loadStereoCalibration(
            params_file_path,
            rotation_vector,
            param_.camera_stereo_params.back().translation_vector,
            camera_ids[i]);
        cv::Rodrigues(rotation_vector, param_.camera_stereo_params.back().rotation_matrix);

        cv::Mat Rt;
        cv::hconcat(
            param_.camera_stereo_params[i].rotation_matrix,
            param_.camera_stereo_params[i].translation_vector,
            Rt);  // [R|t] matrix
        param_.projection_matrices.push_back(
            param_.cameras_intrinsics[i].camera_matrix * Rt);  // Projection matrix
    }
}

cv::Point3f TriangulationNode::triangulatePosition(std::vector<cv::Point2f>& image_points) {
    std::vector<cv::Point2f> point_1 = {image_points[0]};
    std::vector<cv::Point2f> point_2 = {image_points[1]};
    cv::Mat res_point_homogen;
    cv::triangulatePoints(
        param_.projection_matrices[0],
        param_.projection_matrices[1],
        point_1,
        point_2,
        res_point_homogen);
    float x = res_point_homogen.at<float>(0);
    float y = res_point_homogen.at<float>(1);
    float z = res_point_homogen.at<float>(2);
    float w = res_point_homogen.at<float>(3);
    return {x / w, y / w, z / w};
}

}  // namespace handy

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("help:\n./triangulation <path_to_param_file> <path_to_detected_balls.yaml>\n");
    }

    return 0;
}