#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <tuple>

namespace handy {
struct CameraIntrinsicParameters {
    cv::Matx33f camera_matrix;
    cv::Vec<float, 5> dist_coefs;
    std::vector<cv::Mat> rotation_vectors;
    std::vector<cv::Mat> translation_vectors;
    cv::Mat new_camera_matrix;
    std::pair<cv::Mat, cv::Mat> undistort_maps;
};
}  // namespace handy
