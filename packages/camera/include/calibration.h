#pragma once

#include "params.h"
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

#include <atomic>
#include <cstdint>
#include <condition_variable>
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
        kMonoCalibrating = 3,
        kStereoCapturing = 4,
        kStereoCalibrating = 5,
        kOkCalibration = 6
    };

    enum Action { kStart = 1, kCalibrate = 2, kReset = 3 };

  private:
    void declareLaunchParams();
    void initSignals();

    void handleFrame(
        const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg, size_t camera_idx);
    void publishCalibrationState() const;

    void onButtonClick(
        const camera_srvs::srv::CalibrationCommand::Request::SharedPtr& request,
        const camera_srvs::srv::CalibrationCommand::Response::SharedPtr& response);

    void calibrate(size_t camera_idx);
    void stereoCalibrate();
    void handleBadCalibration(size_t camera_idx);
    void handleResetCommand(int camera_idx = -1);
    bool isMonoCalibrated();

    bool checkMaxSimilarity(std::vector<cv::Point2f>& corners, size_t camera_idx) const;
    bool checkEqualFrameNum() const;
    int getImageCoverage(size_t camera_idx) const;

    void initCornerMarkers();
    void appendCornerMarkers(const std::vector<cv::Point2f>& detected_corners, size_t camera_idx);
    visualization_msgs::msg::ImageMarker getCornerMarker(cv::Point2f point);
    visualization_msgs::msg::ImageMarker getBoardMarkerFromCorners(
        std::vector<cv::Point2f>& detected_corners, std_msgs::msg::Header& header);

    struct Signals {
        std::vector<rclcpp::Publisher<foxglove_msgs::msg::ImageMarkerArray>::SharedPtr>
            detected_boards;
        std::vector<rclcpp::Publisher<foxglove_msgs::msg::ImageMarkerArray>::SharedPtr>
            detected_corners;

        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr calibration_state = nullptr;
    } signal_{};

    struct Slots {
        std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> image_sub;
    } slot_{};

    struct Services {
        rclcpp::Service<camera_srvs::srv::CalibrationCommand>::SharedPtr button_service = nullptr;
    } service_{};

    struct Params {
        std::string path_to_params = "";
        std::vector<cv::Point3f> square_obj_points;
        cv::aruco::CharucoBoard charuco_board;

        bool publish_preview_markers = true;

        std::vector<double> marker_color = {0.0, 1.0, 0.0, 0.12};
        double min_accepted_error = 0.75;
        double iou_threshold = 0.5;
        double required_board_coverage = 0.7;

        int64_t fps = 20;
        std::vector<int64_t> cameras_to_calibrate;
        std::map<int, int> id_to_idx;
    } param_;

    struct State {
        std::optional<cv::Size> frame_size = std::nullopt;

        std::vector<std::vector<std::vector<cv::Point2f>>> image_points_all;
        std::vector<std::vector<std::vector<cv::Point3f>>> obj_points_all;
        std::vector<std::vector<Polygon>> polygons_all;
        std::vector<foxglove_msgs::msg::ImageMarkerArray> board_markers_array;
        std::vector<foxglove_msgs::msg::ImageMarkerArray> board_corners_array;

        // unique ID for marker creation
        std::atomic<int> last_marker_id = 0;
        // the number of cameras that currently hold detected charuco corners
        std::atomic<int> currently_detected = 0;
        std::vector<std::atomic<bool>> waiting;
        std::atomic<int> global_calibration_state = kNotCalibrated;
        std::vector<bool> is_mono_calibrated;
        std::condition_variable condvar_to_sync_cameras;
    } state_;

    struct Timer {
        rclcpp::TimerBase::SharedPtr stereo_sync = nullptr;
        rclcpp::TimerBase::SharedPtr calibration_state = nullptr;
    } timer_{};

    struct CallbackGroups {
        rclcpp::CallbackGroup::SharedPtr handle_frame = nullptr;
    } call_group_{};

    std::unique_ptr<cv::aruco::CharucoDetector> charuco_detector_ = nullptr;
    std::vector<CameraIntrinsicParameters> intrinsics_;
};
}  // namespace handy::calibration
