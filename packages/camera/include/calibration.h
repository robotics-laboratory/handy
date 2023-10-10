#pragma once

#include "params.h"
#include "camera_srvs/srv/cmd_service.hpp"

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <visualization_msgs/msg/image_marker.hpp>
#include <foxglove_msgs/msg/image_marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/int16.hpp>
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <vector>
#include <cstdint>
#include <string>
#include <deque>

namespace {

geometry_msgs::msg::Point initPoint(cv::Point2f cv_point);

std::vector<geometry_msgs::msg::Point> getMsgPoints(
    std::vector<cv::Point2f>& detected_corners, cv::Size pattern_size);

}  // namespace
namespace handy::calibration {

typedef boost::geometry::model::d2::point_xy<double> Point;
typedef boost::geometry::model::polygon<Point> Polygon;

const std::vector<Point> getPoints(const std::vector<cv::Point2f>& corners, cv::Size pattern_size);
sensor_msgs::msg::CompressedImage toJpegMsg(const cv_bridge::CvImage& cv_image);

class CalibrationNode : public rclcpp::Node {
  public:
    CalibrationNode();

    enum {
        NOT_CALIBRATED = 1,
        CAPTURING = 2,
        CALIBRATING = 3,
        BAD_CALIBRATION = 4,
        OK_CALIBRATION = 5,

        START = 6,
        CALIBRATE = 7,
        RESET = 8
    };

  private:
    void declareLaunchParams();
    void initSignals();

    void handleFrame(const sensor_msgs::msg::CompressedImage::ConstPtr& msg);
    void publishCalibrationState() const;

    void onButtonClick(
        const camera_srvs::srv::CmdService::Request::SharedPtr request,
        camera_srvs::srv::CmdService::Response::SharedPtr response);

    void calibrate();
    void saveCalibParams() const;

    bool checkMaxSimilarity(std::vector<cv::Point2f> corners) const;

    visualization_msgs::msg::ImageMarker getBoardMarkerFromCorners(
        std::vector<cv::Point2f>& detected_corners, cv_bridge::CvImagePtr image_ptr);

    struct Signals {
        rclcpp::Publisher<foxglove_msgs::msg::ImageMarkerArray>::SharedPtr detected_boards =
            nullptr;
        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr calibration_state = nullptr;
        rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr chessboard_preview_pub =
            nullptr;
    } signal_{};

    struct Slots {
        rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub = nullptr;
    } slot_{};

    struct Services {
        rclcpp::Service<camera_srvs::srv::CmdService>::SharedPtr button_service = nullptr;
    } service_{};

    struct Params {
        std::string path_to_save_params = "param_save";
        cv::Size pattern_size = cv::Size(9, 6);

        std::vector<cv::Point3f> square_obj_points;

        bool publish_chessboard_preview = true;
        bool auto_calibrate = true;

        std::vector<double> marker_color = {0.0, 1.0, 0.0};
        double min_accepted_error = 0.75;
        double alpha_chn_increase = 0.12;
        double IoU_treshhold = 0.5;
    } param_;

    struct State {
        cv::Size frame_size_ = cv::Size(0, 0);

        std::vector<std::vector<cv::Point2f>> detected_corners_all;
        std::vector<std::vector<cv::Point3f>> object_points_all;
        foxglove_msgs::msg::ImageMarkerArray board_markers_array;

        int last_marker_id = 0;
        int calibration_state = NOT_CALIBRATED;
    } state_;

    CameraIntrinsicParameters intrinsic_params_{};
};
}  // namespace handy::calibration