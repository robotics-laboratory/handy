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
    bool handleFrame(const cv::Mat& image, int camera_idx);
    void calibrate(int camera_idx);
    void stereoCalibrate();
    void clearDetections();
    void clearLastDetection(int camera_idx);

  private:
    void declareLaunchParams(const YAML::Node& param_node);

    bool checkMaxSimilarity(std::vector<std::vector<cv::Point2f>>& corners, int camera_idx) const;
    int getImageCoverage(int camera_idx) const;
    void fillImageObjectPoints(
        std::vector<std::vector<cv::Point2f>>& image_points,
        std::vector<std::vector<cv::Point3f>>& obj_points, int camera_idx);

    void initCornerMarkers();
    void appendCornerMarkers(const std::vector<cv::Point2f>& detected_corners);
    // visualization_msgs::msg::ImageMarker getCornerMarker(cv::Point2f point);
    // visualization_msgs::msg::ImageMarker getBoardMarkerFromCorners(
    // std::vector<cv::Point2f>& detected_corners, std_msgs::msg::Header& header);

    struct Params {
        int camera_num = 2;
        std::string path_to_params = "";
        std::vector<cv::Point3f> square_obj_points;
        cv::aruco::CharucoBoard charuco_board;
        cv::aruco::GridBoard aruco_board;

        bool publish_preview_markers = true;

        std::vector<double> marker_color = {0.0, 1.0, 0.0, 0.12};
        double min_accepted_error = 0.75;
        double iou_threshold = 0.5;
        double required_board_coverage = 0.7;
        std::vector<int> camera_ids;
    } param_;

    struct State {
        std::optional<cv::Size> frame_size = std::nullopt;

        // for each camera, each view, each detected marker vector of 4 aruco corners
        std::vector<std::vector<std::vector<std::vector<cv::Point2f>>>> marker_corners_all;
        std::vector<std::vector<std::vector<int>>> marker_ids_all;
        std::vector<std::vector<Polygon>> polygons_all;

        std::vector<bool> is_calibrated;
        std::vector<CameraIntrinsicParameters> intrinsic_params;
    } state_;

    std::unique_ptr<cv::aruco::CharucoDetector> charuco_detector_ = nullptr;
    std::unique_ptr<cv::aruco::ArucoDetector> aruco_detector_ = nullptr;
};
}  // namespace handy::calibration
