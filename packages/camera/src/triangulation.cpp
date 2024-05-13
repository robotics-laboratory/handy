#include "triangulation.h"

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/calib3d.hpp>
// #include <opencv2/imgproc.hpp>
#include <algorithm>
#include <fstream>
// #include <opencv2/sfm/projection.hpp>

namespace handy {
TriangulationNode::TriangulationNode(std::string& params_file_path, std::vector<int> camera_ids) {
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

cv::Point2f normalize(const cv::Point2f& point, const cv::Mat& camera_matrix) {
    cv::Point2f normalized_point;
    normalized_point.x =
        (point.x - camera_matrix.at<double>(0, 2)) / camera_matrix.at<double>(0, 0);
    normalized_point.y =
        (point.y - camera_matrix.at<double>(1, 2)) / camera_matrix.at<double>(1, 1);
    return normalized_point;
}

cv::Point3f TriangulationNode::triangulatePosition(std::vector<cv::Point2f> image_points) {
    // https://en.wikipedia.org/wiki/Triangulation_(computer_vision)
    // std::vector<cv::Point2f> point_1 = {image_points[0]};
    // std::vector<cv::Point2f> point_2 = {image_points[1]};
    // std::vector<cv::Point2f> undistorted_point_1;
    // std::vector<cv::Point2f> undistorted_point_2;
    // cv::undistortPoints(
    //     point_1,
    //     undistorted_point_1,
    //     param_.cameras_intrinsics[0].camera_matrix,
    //     param_.cameras_intrinsics[0].dist_coefs,
    //     cv::noArray(),
    //     param_.cameras_intrinsics[0].camera_matrix);
    // cv::undistortPoints(
    //     point_2,
    //     undistorted_point_2,
    //     param_.cameras_intrinsics[1].camera_matrix,
    //     param_.cameras_intrinsics[1].dist_coefs,
    //     cv::noArray(),
    //     param_.cameras_intrinsics[1].camera_matrix);

    // undistorted_point_1[0] =
    //     normalize(undistorted_point_1[0], param_.cameras_intrinsics[0].camera_matrix);
    // undistorted_point_2[0] =
    //     normalize(undistorted_point_2[0], param_.cameras_intrinsics[1].camera_matrix);

    // // std::vector<cv::Point2f> undistorted_point_1 = point_1;
    // // std::vector<cv::Point2f> undistorted_point_2 = point_2;

    // // cv::undistortPoints(
    // //     point_1,
    // //     undistorted_point_1,
    // //     param_.cameras_intrinsics[0].camera_matrix,
    // //     param_.cameras_intrinsics[0].dist_coefs,
    // //     cv::noArray(),
    // //     param_.projection_matrices[0]);
    // // printf(
    // //     "%f %f %f %f\n",
    // //     point_1[0].x,
    // //     point_1[0].y,
    // //     undistorted_point_1[0].x,
    // //     undistorted_point_1[0].y);

    // // cv::undistortPoints(
    // //     point_2,
    // //     undistorted_point_2,
    // //     param_.cameras_intrinsics[1].camera_matrix,
    // //     param_.cameras_intrinsics[1].dist_coefs,
    // //     cv::noArray(),
    // //     param_.projection_matrices[1]);

    // float up = (param_.camera_stereo_params[1].rotation_matrix.row(0) -
    //             undistorted_point_2[0].x * param_.camera_stereo_params[1].rotation_matrix.row(2))
    //                .dot(param_.camera_stereo_params[1].translation_vector.t());
    // // printf("up: %f\n", up);
    // float down =
    //     (param_.camera_stereo_params[1].rotation_matrix.row(0) -
    //      undistorted_point_2[0].x * param_.camera_stereo_params[1].rotation_matrix.row(2))
    //         .dot(
    //             (cv::Mat_<double>(1, 3) << undistorted_point_1[0].x, undistorted_point_1[0].y, 1.));
    // // printf("down: %f\n", down);

    // float x_3 = up / down;
    // cv::Point3f result_point{undistorted_point_1[0].x * x_3, undistorted_point_1[0].y * x_3, x_3};
    // return result_point;

    ////////////////////////////////////////////////////
    // std::vector<cv::Point2f> point_1 =
    // {param_.cameras_intrinsics[0].toImageCoord(image_points[0])}; std::vector<cv::Point2f>
    // point_2 = {param_.cameras_intrinsics[1].toImageCoord(image_points[1])};

    std::vector<cv::Point2f> point_1 = {image_points[0]};
    std::vector<cv::Point2f> point_2 = {image_points[1]};

    std::vector<cv::Point2f> undistorted_point_1 = point_1;
    std::vector<cv::Point2f> undistorted_point_2 = point_2;
    // cv::undistortPoints(
    //     point_1,
    //     undistorted_point_1,
    //     param_.cameras_intrinsics[0].camera_matrix,
    //     param_.cameras_intrinsics[0].dist_coefs,
    //     cv::noArray(),
    //     param_.projection_matrices[0]);
    // cv::undistortPoints(
    //     point_2,
    //     undistorted_point_2,
    //     param_.cameras_intrinsics[1].camera_matrix,
    //     param_.cameras_intrinsics[1].dist_coefs,
    //     cv::noArray(),
    //     param_.projection_matrices[1]);

    cv::Mat res_point_homogen;
    cv::triangulatePoints(
        param_.projection_matrices[0],
        param_.projection_matrices[1],
        undistorted_point_1,
        undistorted_point_2,
        res_point_homogen);
    float x = res_point_homogen.at<float>(0);
    float y = res_point_homogen.at<float>(1);
    float z = res_point_homogen.at<float>(2);
    float w = res_point_homogen.at<float>(3);
    printf("%f %f %f %f\n", x, y, z, w);
    printf("%f %f %f\n", x/w, y/w, z/w);
    printf("%f %f %f %f\n\n", image_points[0].x, image_points[0].y, image_points[1].x,
    image_points[1].y); return {x / w, y / w, z / w};
}

}  // namespace handy

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("help:\n./triangulation <path_to_param_file> <path_to_detected_balls.yaml>\n");
        return 0;
    }
    YAML::Node detections_yaml = YAML::LoadFile(argv[2]);
    auto first_camera = detections_yaml
                            ["/home/bakind/handy/datasets/TableOrange2msRecord_22_04/"
                             "orange_dark_2ms/orange_dark_2ms_2_1_mask"]["bounding_boxes"]
                                .as<std::vector<std::vector<double>>>();
    auto second_camera = detections_yaml
                             ["/home/bakind/handy/datasets/TableOrange2msRecord_22_04/"
                              "orange_dark_2ms/orange_dark_2ms_2_2_mask"]["bounding_boxes"]
                                 .as<std::vector<std::vector<double>>>();

    if (first_camera.size() != second_camera.size()) {
        printf(
            "number of frames does not match: %ld and %ld\n",
            first_camera.size(),
            second_camera.size());
        exit(EXIT_FAILURE);
    }
    printf("%d\n", first_camera.size());
    std::string path_file_param(argv[1]);
    handy::TriangulationNode triangulation_node(path_file_param, {1, 2});
    std::vector<cv::Point3f> triangulated_points;

    detections_yaml["triangulated_points"] = YAML::Node(YAML::NodeType::Sequence);
    detections_yaml["triangulated_points"].SetStyle(YAML::EmitterStyle::Flow);
    detections_yaml["detected_points"] = YAML::Node(YAML::NodeType::Sequence);
    detections_yaml["detected_points"].SetStyle(YAML::EmitterStyle::Flow);

    for (int i = 0; i < first_camera.size(); ++i) {
        cv::Point2f first_image_point{
            static_cast<float>(first_camera[i][2] + first_camera[i][0]) / 2,
            static_cast<float>(first_camera[i][3] + first_camera[i][1]) / 2};
        cv::Point2f second_image_point{
            static_cast<float>(second_camera[i][2] + second_camera[i][0]) / 2,
            static_cast<float>(second_camera[i][3] + second_camera[i][1]) / 2};

        triangulated_points.push_back(
            triangulation_node.triangulatePosition({first_image_point, second_image_point}));

        std::vector<float> vector_point = {
            triangulated_points.back().x,
            triangulated_points.back().y,
            triangulated_points.back().z};
        detections_yaml["triangulated_points"].push_back(vector_point);
        std::vector<std::vector<float>> detected_points = {
            {first_image_point.x, first_image_point.y},
            {second_image_point.x, second_image_point.y}};
        detections_yaml["detected_points"].push_back(detected_points);
    }
    YAML::Emitter out;
    out << detections_yaml;
    std::ofstream fout(argv[2]);
    fout << out.c_str();

    YAML::Node params_yaml = YAML::LoadFile(argv[1]);
    first_camera =
        params_yaml["parameters"]["1"]["common_points"].as<std::vector<std::vector<double>>>();
    second_camera =
        params_yaml["parameters"]["2"]["common_points"].as<std::vector<std::vector<double>>>();

    std::vector<cv::Point3f> triangulated_common_points;

    params_yaml["triangulated_common_points"] = YAML::Node(YAML::NodeType::Sequence);
    params_yaml["triangulated_common_points"].SetStyle(YAML::EmitterStyle::Flow);

    for (int i = 0; i < first_camera.size(); ++i) {
        cv::Point2f first_image_point{first_camera[i][0], first_camera[i][1]};
        cv::Point2f second_image_point{second_camera[i][0], second_camera[i][1]};

        triangulated_common_points.push_back(
            triangulation_node.triangulatePosition({first_image_point, second_image_point}));

        std::vector<float> vector_point = {
            triangulated_common_points.back().x,
            triangulated_common_points.back().y,
            triangulated_common_points.back().z};
        params_yaml["triangulated_common_points"].push_back(vector_point);
    }

    YAML::Emitter params_out;
    params_out << params_yaml;
    std::ofstream fout_params_common(argv[1]);
    if (!fout_params_common.is_open()) {
        perror("file");
    }
    fout_params_common << params_out.c_str();

    return 0;
}