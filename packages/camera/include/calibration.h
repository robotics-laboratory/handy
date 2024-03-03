#pragma once

#include "camera_srvs/srv/calibration_command.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <foxglove_msgs/msg/image_marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/int16.hpp>
#include <visualization_msgs/msg/image_marker.hpp>

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core/core.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace handy::calibration {

using Point = boost::geometry::model::d2::point_xy<double>;
using Polygon = boost::geometry::model::polygon<Point>;
using MultiPolygon = boost::geometry::model::multi_polygon<Polygon>;

double getIou(const Polygon& first, const Polygon& second);

const std::vector<Point> toBoostCorners(
    const std::vector<cv::Point2f>& corners, cv::Size pattern_size);
sensor_msgs::msg::CompressedImage toJpegMsg(const cv_bridge::CvImage& cv_image);

class CalibrationNode : public rclcpp::Node {
  public:
    CalibrationNode();

    enum CalibrationState {
        kNotCalibrated = 1,
        kCapturing = 2,
        kCalibrating = 3,
        kOkCalibration = 5
    };

    enum Action { kStart = 1, kCalibrate = 2, kReset = 3 };

  private:
    void declareLaunchParams();
    void initSignals();

    void handleFrame(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);
    void publishCalibrationState() const;

    void onButtonClick(
        const camera_srvs::srv::CalibrationCommand::Request::SharedPtr& request,
        const camera_srvs::srv::CalibrationCommand::Response::SharedPtr& response);

    void calibrate();
    void handleBadCalibration();
    void handleResetCommand();

    bool checkMaxSimilarity(std::vector<cv::Point2f>& corners) const;
    int getImageCoverage() const;

    void initCornerMarkers();
    void appendCornerMarkers(const std::vector<cv::Point2f>& detected_corners);
    visualization_msgs::msg::ImageMarker getCornerMarker(cv::Point2f point);
    visualization_msgs::msg::ImageMarker getBoardMarkerFromCorners(
        std::vector<cv::Point2f>& detected_corners, std_msgs::msg::Header& header);

    struct Signals {
        rclcpp::Publisher<foxglove_msgs::msg::ImageMarkerArray>::SharedPtr detected_boards =
            nullptr;
        rclcpp::Publisher<foxglove_msgs::msg::ImageMarkerArray>::SharedPtr detected_corners =
            nullptr;

        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr calibration_state = nullptr;
    } signal_{};

    struct Slots {
        rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub = nullptr;
    } slot_{};

    struct Services {
        rclcpp::Service<camera_srvs::srv::CalibrationCommand>::SharedPtr button_service = nullptr;
    } service_{};

    struct Params {
        std::string path_to_save_params = "";
        std::vector<cv::Point3f> square_obj_points;
        cv::aruco::CharucoBoard charuco_board;

        bool publish_preview_markers = true;
        bool auto_calibrate = true;

        std::vector<double> marker_color = {0.0, 1.0, 0.0, 0.12};
        double min_accepted_error = 0.75;
        double iou_threshold = 0.5;
        double required_board_coverage = 0.7;
    } param_;

    struct State {
        std::optional<cv::Size> frame_size = std::nullopt;

        std::vector<std::vector<cv::Point2f>> image_points_all;
        std::vector<std::vector<cv::Point3f>> obj_points_all;
        std::vector<Polygon> polygons_all;
        foxglove_msgs::msg::ImageMarkerArray board_markers_array;
        foxglove_msgs::msg::ImageMarkerArray board_corners_array;

        int last_marker_id = 0;
        int calibration_state = kNotCalibrated;
    } state_;

    struct Timer {
        rclcpp::TimerBase::SharedPtr calibration_state = nullptr;
    } timer_{};

    std::unique_ptr<cv::aruco::CharucoDetector> charuco_detector_ = nullptr;
};
}  // namespace handy::calibration
