#include "params.h"
#include "params.h"

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace handy {
// the following interface counts on 2 cameras only

struct StereoParameters {
    cv::Mat rotation_matrix;
    cv::Mat translation_vector;
};

class TriangulationNode {
  public:
    TriangulationNode(std::string& params_file_path, std::vector<int> camera_ids);

    cv::Point3f triangulatePosition(std::vector<cv::Point2f> image_points);

  private:
    struct Params {
        std::vector<CameraIntrinsicParameters> cameras_intrinsics = {};
        std::vector<StereoParameters> camera_stereo_params = {};
        std::vector<cv::Mat> projection_matrices = {};
    } param_{};
};
}  // namespace handy
