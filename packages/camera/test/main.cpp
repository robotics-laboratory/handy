#include <gtest/gtest.h>

#include "params.h"
#include <opencv2/core/core.hpp>

TEST(camera, params_read_write) {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1., 0., 1., 0., -2., 0., 3., 0., 3.);
    cv::Vec<float, 5> distort_coefs(1, 2, 3, 4, 5);
    cv::Size image_size(1920, 1080);

    handy::CameraIntrinsicParameters params(image_size, camera_matrix, distort_coefs);
    
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}