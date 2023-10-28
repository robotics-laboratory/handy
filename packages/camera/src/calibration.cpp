#include "calibration.h"

#include <opencv2/calib3d.hpp>
#include <deque>

namespace {

/**
 * Converts OpenCV point to ROS2 message
 *
 * @param cv_point OpenCV point that will be converted
 * @return converted point of ROS2 message
 */
geometry_msgs::msg::Point toMsgPoint(cv::Point2f cv_point) {
    geometry_msgs::msg::Point point;
    point.x = cv_point.x;
    point.y = cv_point.y;
    return point;
}

/**
 * Converts all detected corners into a vector of top/bottom left/right corners
 *
 * @param cv_point OpenCV point that will be converted
 * @return vector of 4 corners of detected pattern
 */
std::vector<geometry_msgs::msg::Point> toMsgCorners(
    const std::vector<cv::Point2f>& detected_corners, cv::Size pattern_size) {
    cv::Point2f board_cv_corners[4] = {
        detected_corners[0],
        detected_corners[pattern_size.width - 1],
        detected_corners[pattern_size.width * pattern_size.height - 1],
        detected_corners[pattern_size.width * (pattern_size.height - 1)],
    };

    std::vector<geometry_msgs::msg::Point> marker_points(4);
    for (int i = 0; i < 4; ++i) {
        marker_points[i] = toMsgPoint(board_cv_corners[i]);
    }
    return marker_points;
}

}  // namespace
namespace handy::calibration {

using namespace std::chrono_literals;
using namespace boost::geometry;

/**
 * Converts all corners of detected chessboard into a closed polyline of
 * top/bottom left/right corners (Boost points)
 *
 * @param corners vector all corners in cv::Point2f format
 * @param pattern_size width and height of chessboard
 * @return vector of points of closed polyline
 */
const std::vector<Point> convertToBoostCorners(
    const std::vector<cv::Point2f>& corners, cv::Size pattern_size) {
    // clang-format off
    const std::vector<cv::Point2f> cv_points = {
        corners[0],                                               // top left
        corners[pattern_size.width - 1],                          // top right
        corners[pattern_size.width * pattern_size.height - 1],    // bottom right
        corners[pattern_size.width * (pattern_size.height - 1)],  // bottom left
    };
    // clang-format on

    const std::vector<Point> points = {
        {cv_points[0].x, cv_points[0].y},  // top left
        {cv_points[3].x, cv_points[3].y},  // top right
        {cv_points[2].x, cv_points[2].y},  // bottom right
        {cv_points[1].x, cv_points[1].y},  // bottom left
        {cv_points[0].x, cv_points[0].y}};

    return points;
}

/**
 * Calculates "intersection over union" metric as a measure of similarity between two polygons
 *
 * @param first Boost polygon
 * @param second Boost polygon
 * @return double value of metric
 */

double getIou(const Polygon& first, const Polygon& second) {
    std::deque<Polygon> inter_poly;
    intersection(first, second, inter_poly);

    double iou = 0.0;
    if (inter_poly.size() > 1) {
        throw std::logic_error("Intersection is not valid");
    }
    if (!inter_poly.empty()) {
        double inter_area = area(inter_poly.front());
        iou = inter_area / (area(first) + area(second) - inter_area);
    }
    return iou;
}

sensor_msgs::msg::CompressedImage toJpegMsg(const cv_bridge::CvImage& cv_image) {
    sensor_msgs::msg::CompressedImage result;
    cv_image.toCompressedImageMsg(result, cv_bridge::Format::JPEG);
    return result;
}

CalibrationNode::CalibrationNode() : Node("calibration_node") {
    RCLCPP_INFO_STREAM(this->get_logger(), "Init started");

    declareLaunchParams();
    RCLCPP_INFO_STREAM(this->get_logger(), "Launch params declared");

    intrinsic_params_ = CameraIntrinsicParameters(param_.path_to_save_params, "camera_0");

    initSignals();
    RCLCPP_INFO_STREAM(this->get_logger(), "Signals initialised");

    for (int i = 0; i < param_.pattern_size.height; ++i) {
        for (int j = 0; j < param_.pattern_size.width; ++j) {
            param_.square_obj_points.push_back(cv::Point3f(i, j, 0));
        }
    }
    publishCalibrationState();
    signal_.detected_boards->publish(state_.board_markers_array);
}

void CalibrationNode::declareLaunchParams() {
    // clang-format off
    param_.publish_preview_markers = this->declare_parameter<bool>("publish_preview_markers", true);
    param_.path_to_save_params = this->declare_parameter<std::string>("calibration_file_path", "param_save");

    param_.coverage_required = this->declare_parameter<double>("coverage_required", 0.7);
    param_.min_accepted_error = this->declare_parameter<double>("min_accepted_calib_error", 0.75);
    param_.iou_treshhold = this->declare_parameter<double>("iou_threshold", 0.5);
    param_.marker_color = this->declare_parameter<std::vector<double>>("marker_color", {0.0f, 1.0f, 0.0f, 0.12f});
    // clang-format on
}

void CalibrationNode::initSignals() {
    slot_.image_sub = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/camera_1/bgr/image",
        10,
        std::bind(&CalibrationNode::handleFrame, this, std::placeholders::_1));
    service_.button_service = this->create_service<camera_srvs::srv::CmdService>(
        "/calibration_1/button",
        std::bind(
            &CalibrationNode::onButtonClick, this, std::placeholders::_1, std::placeholders::_2));

    signal_.detected_boards = this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>(
        "/calibration_1/board_markers", 10);
    signal_.detected_corners = this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>(
        "/calibration_1/corner_markers", 10);

    signal_.calibration_state =
        this->create_publisher<std_msgs::msg::Int16>("/calibration_1/state", 10);
}

void CalibrationNode::handleFrame(const sensor_msgs::msg::CompressedImage::ConstPtr& msg) {
    if (state_.calibration_state != CAPTURING) {
        return;
    }

    cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    if (!state_.frame_size_) {
        state_.frame_size_ =
            std::make_optional<cv::Size>(image_ptr->image.cols, image_ptr->image.rows);
    }
    assert(
        state_.frame_size_->width == image_ptr->image.cols &&
        state_.frame_size_->height == image_ptr->image.rows);
    std::vector<cv::Point2f> detected_corners;
    bool found = cv::findChessboardCorners(
        image_ptr->image, param_.pattern_size, detected_corners, cv::CALIB_CB_FAST_CHECK);
    if (found && checkMaxSimilarity(detected_corners)) {
        if (param_.publish_preview_markers) {
            state_.board_markers_array.markers.push_back(
                getBoardMarkerFromCorners(detected_corners, image_ptr->header));
            signal_.detected_boards->publish(state_.board_markers_array);
        }

        state_.detected_corners_all.push_back(detected_corners);
        state_.object_points_all.push_back(param_.square_obj_points);
        state_.polygons_all.emplace_back();

        const std::vector<Point> cur_points =
            convertToBoostCorners(detected_corners, param_.pattern_size);
        append(state_.polygons_all.back(), cur_points);
    }
    state_.board_corners_array.markers.clear();
    if (param_.publish_preview_markers && found) {
        appendCornerMarkers(detected_corners);
    }
    publishCalibrationState();
    signal_.detected_corners->publish(state_.board_corners_array);
}

void CalibrationNode::onButtonClick(
    const camera_srvs::srv::CmdService::Request::SharedPtr request,
    camera_srvs::srv::CmdService::Response::SharedPtr) {
    if (state_.calibration_state == CALIBRATING) {
        return;
    }
    if (request->cmd == START) {
        state_.calibration_state = CAPTURING;
        RCLCPP_INFO_STREAM(this->get_logger(), "State set to capturing");

    } else if (request->cmd == CALIBRATE) {
        calibrate();
    } else if (request->cmd == RESET) {
        state_.calibration_state = NOT_CALIBRATED;
        state_.detected_corners_all.clear();
        state_.object_points_all.clear();
        state_.board_markers_array.markers.clear();
        state_.polygons_all.clear();
        signal_.detected_boards->publish(state_.board_markers_array);
    }
    publishCalibrationState();
}

bool CalibrationNode::checkMaxSimilarity(std::vector<cv::Point2f>& corners) const {
    Polygon new_poly;
    const std::vector<Point> new_points = convertToBoostCorners(corners, param_.pattern_size);
    append(new_poly, new_points);

    return std::all_of(
        state_.polygons_all.begin(), state_.polygons_all.end(), [&](const Polygon& prev_poly) {
            return getIou(prev_poly, new_poly) <= param_.iou_treshhold;
        });
}

int CalibrationNode::checkOverallCoverage() const {
    MultiPolygon current_coverage;
    for (const Polygon& new_poly : state_.polygons_all) {
        MultiPolygon tmp_poly;
        union_(current_coverage, new_poly, tmp_poly);
        current_coverage = tmp_poly;
    }
    float ratio = area(current_coverage) / (state_.frame_size_->height * state_.frame_size_->width);
    return static_cast<int>(ratio * 100);
}

void CalibrationNode::publishCalibrationState() const {
    std_msgs::msg::Int16 msg;
    msg.data = state_.calibration_state;
    signal_.calibration_state->publish(msg);
}

visualization_msgs::msg::ImageMarker CalibrationNode::getBoardMarkerFromCorners(
    std::vector<cv::Point2f>& detected_corners, std_msgs::msg::Header& header) {
    visualization_msgs::msg::ImageMarker marker;

    marker.header.frame_id = header.frame_id;
    marker.header.stamp = header.stamp;
    marker.ns = "calibration";
    marker.id = ++state_.last_marker_id;
    marker.type = visualization_msgs::msg::ImageMarker::POLYGON;
    marker.action = visualization_msgs::msg::ImageMarker::ADD;

    marker.position.x = 0;
    marker.position.y = 0;

    marker.scale = 1.;

    marker.filled = 1;
    marker.fill_color.r = param_.marker_color[0];
    marker.fill_color.g = param_.marker_color[1];
    marker.fill_color.b = param_.marker_color[2];
    marker.fill_color.a = param_.marker_color[3];

    marker.points = toMsgCorners(detected_corners, param_.pattern_size);
    return marker;
}
visualization_msgs::msg::ImageMarker CalibrationNode::getCornerMarker(cv::Point2f point) {
    visualization_msgs::msg::ImageMarker marker;

    marker.ns = "calibration";
    marker.id = ++state_.last_marker_id;
    marker.type = visualization_msgs::msg::ImageMarker::CIRCLE;
    marker.action = visualization_msgs::msg::ImageMarker::ADD;

    marker.position.x = point.x;
    marker.position.y = point.y;

    marker.scale = 6;

    marker.filled = 1;
    marker.fill_color.r = 1. - param_.marker_color[0];
    marker.fill_color.g = 1. - param_.marker_color[1];
    marker.fill_color.b = 1. - param_.marker_color[2];
    marker.fill_color.a = 0.7;

    return marker;
}

void CalibrationNode::appendCornerMarkers(const std::vector<cv::Point2f>& detected_corners) {
    for (size_t i = 0; i < detected_corners.size(); ++i) {
        visualization_msgs::msg::ImageMarker marker = getCornerMarker(detected_corners[i]);
        state_.board_corners_array.markers.push_back(marker);
    }
}

void CalibrationNode::handleBadCalibration() {
    RCLCPP_INFO_STREAM(this->get_logger(), "Continue taking frames");

    state_.calibration_state = CAPTURING;
    publishCalibrationState();
}

void CalibrationNode::calibrate() {
    state_.calibration_state = CALIBRATING;
    publishCalibrationState();
    RCLCPP_INFO_STREAM(this->get_logger(), "Calibration inialized");
    int coverage_percentage = checkOverallCoverage();
    if (coverage_percentage < param_.coverage_required) {
        RCLCPP_INFO(this->get_logger(), "Coverage of %d is not sufficient", coverage_percentage);
        handleBadCalibration();
        return;
    }
    std::vector<cv::Mat> _1, _2;
    double rms = cv::calibrateCamera(
        state_.object_points_all,
        state_.detected_corners_all,
        *state_.frame_size_,
        intrinsic_params_.camera_matrix,
        intrinsic_params_.dist_coefs,
        _1,
        _2);

    RCLCPP_INFO(
        this->get_logger(),
        "Calibration done with error of %f and coverage of %d%%",
        rms,
        coverage_percentage);
    if (rms < param_.min_accepted_error) {
        saveCalibParams();
        state_.calibration_state = OK_CALIBRATION;
    } else {
        handleBadCalibration();
        return;
    }
    publishCalibrationState();
}

void CalibrationNode::saveCalibParams() const {
    intrinsic_params_.save();
}
}  // namespace handy::calibration