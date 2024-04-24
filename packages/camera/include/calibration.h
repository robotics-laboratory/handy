#pragma once

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core/core.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include <params.h>


namespace handy::calibration {

using Point = boost::geometry::model::d2::point_xy<double>;
using Polygon = boost::geometry::model::polygon<Point>;
using MultiPolygon = boost::geometry::model::multi_polygon<Polygon>;

double getIou(const Polygon& first, const Polygon& second);

const std::vector<Point> toBoostCorners(
    const std::vector<cv::Point2f>& corners, cv::Size pattern_size);

class CalibrationNode {
  public:
    CalibrationNode(const YAML::Node& param_node);

  private:
    void declareLaunchParams(const YAML::Node& param_node);

    bool handleFrame(const cv::Mat& image, int camera_idx);

    void calibrate(int camera_idx);
    void handleBadCalibration();
    void handleResetCommand();

    bool checkMaxSimilarity(std::vector<cv::Point2f>& corners) const;
    int getImageCoverage() const;

    void initCornerMarkers();
    void appendCornerMarkers(const std::vector<cv::Point2f>& detected_corners);
    // visualization_msgs::msg::ImageMarker getCornerMarker(cv::Point2f point);
    // visualization_msgs::msg::ImageMarker getBoardMarkerFromCorners(
        // std::vector<cv::Point2f>& detected_corners, std_msgs::msg::Header& header);

    struct Params {
        int camera_num = 2;
        std::string path_to_save_params = "";
        std::vector<cv::Point3f> square_obj_points;
        cv::aruco::CharucoBoard charuco_board;

        bool publish_preview_markers = true;

        std::vector<double> marker_color = {0.0, 1.0, 0.0, 0.12};
        double min_accepted_error = 0.75;
        double iou_threshold = 0.5;
        double required_board_coverage = 0.7;
    } param_;

    struct State {
        std::optional<cv::Size> frame_size = std::nullopt;

        std::vector<std::vector<std::vector<cv::Point2f>>> image_points_all;
        std::vector<std::vector<std::vector<cv::Point3f>>> obj_points_all;
        std::vector<std::vector<Polygon>> polygons_all;

        std::vector<bool> is_calibrated;
        std::vector<CameraIntrinsicParameters> is_calibrated;
    } state_;

    std::unique_ptr<cv::aruco::CharucoDetector> charuco_detector_ = nullptr;
};
}  // namespace handy::calibration
