#include <gtest/gtest.h>

#include "params.h"
#include <opencv2/core/core.hpp>
#include <fstream>
#include <string>

TEST(camera, params_single_read_write) {
    const std::string path_to_yaml = "test_params.yaml";
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1., 0., 1., 0., -2., 0., 3., 0., 3.);
    cv::Vec<float, 5> distort_coefs(1, 2, 3, 4, 5);
    cv::Size image_size(1920, 1080);

    handy::CameraIntrinsicParameters params_1(image_size, camera_matrix, distort_coefs, 0);
    params_1.storeYaml(path_to_yaml);

    std::ifstream test_file(path_to_yaml);
    EXPECT_TRUE(test_file);  // opened successfully
    test_file.close();

    handy::CameraIntrinsicParameters params_1_check =
        handy::CameraIntrinsicParameters::loadFromYaml(path_to_yaml, 0);
    EXPECT_EQ(image_size, params_1_check.image_size);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(
                camera_matrix.at<double>(i, j), params_1_check.camera_matrix.at<double>(i, j));
        }
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(distort_coefs[i], params_1_check.dist_coefs[i]);
    }
}

TEST(camera, params_multiple_read_write) {
    // create test params
    const std::string path_to_yaml = "test_params.yaml";
    cv::Mat camera_matrix_1 = (cv::Mat_<double>(3, 3) << 0., 0., 1., 0., -2., 0., 3., 0., 3.);
    cv::Mat camera_matrix_2 = (cv::Mat_<double>(3, 3) << 42., 0., 1., 0.4242, -2., 0., 3., 0., 3.);
    cv::Vec<float, 5> distort_coefs(1, 2, 3, 4, 5);
    cv::Size image_size(1920, 1080);

    // store two IDs into the same file
    handy::CameraIntrinsicParameters params_1(image_size, camera_matrix_1, distort_coefs, 0);
    params_1.storeYaml(path_to_yaml);

    handy::CameraIntrinsicParameters params_2(image_size, camera_matrix_2, distort_coefs, 1);
    params_2.storeYaml(path_to_yaml);

    // check for file to exist
    std::ifstream test_file(path_to_yaml);
    EXPECT_TRUE(test_file);  // opened successfully
    test_file.close();

    // loading and checking all params for the first ID
    handy::CameraIntrinsicParameters params_1_check =
        handy::CameraIntrinsicParameters::loadFromYaml(path_to_yaml, 0);
    EXPECT_EQ(image_size, params_1_check.image_size);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(
                camera_matrix_1.at<double>(i, j), params_1_check.camera_matrix.at<double>(i, j));
        }
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(distort_coefs[i], params_1_check.dist_coefs[i]);
    }

    // loading and checking all params for the second ID
    handy::CameraIntrinsicParameters params_2_check =
        handy::CameraIntrinsicParameters::loadFromYaml(path_to_yaml, 1);
    EXPECT_EQ(image_size, params_2_check.image_size);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(
                camera_matrix_2.at<double>(i, j), params_2_check.camera_matrix.at<double>(i, j));
        }
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(distort_coefs[i], params_2_check.dist_coefs[i]);
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}