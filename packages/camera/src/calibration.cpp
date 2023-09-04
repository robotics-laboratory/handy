#include "calibration.h"

namespace {

geometry_msgs::msg::Point initPoint(cv::Point2f cv_point) {
    geometry_msgs::msg::Point point;
    point.x = cv_point.x;
    point.y = cv_point.y;
    return point;
}

}  // namespace
namespace handy::calibration {

using namespace std::chrono_literals;

const std::vector<Point> getPoints(const std::vector<cv::Point2f> corners, cv::Size pattern_size) {
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

CalibrationNode::CalibrationNode() : Node("calibration_node") {
    RCLCPP_INFO_STREAM(this->get_logger(), "Init started");

    declareLaunchParams();
    RCLCPP_INFO_STREAM(this->get_logger(), "Launch params declared");

    initSignals();
    RCLCPP_INFO_STREAM(this->get_logger(), "Signals initialised");

    for (int i = 0; i < params_.pattern_size.height; ++i) {
        for (int j = 0; j < params_.pattern_size.width; ++j) {
            params_.square_obj_points.push_back(cv::Point3f(i, j, 0));
        }
    }
}

void CalibrationNode::declareLaunchParams() {
    // clang-format off
    params_.publish_chessboard_preview = this->declare_parameter<bool>("publish_chessboard_preview", true);
    params_.auto_calibrate = this->declare_parameter<bool>("auto_calibrate", true);
    params_.path_to_save_params = this->declare_parameter<std::string>("calibration_file_path", "param_save");

    params_.min_accepted_error = this->declare_parameter<double>("min_accepted_calib_error", 0.75);
    params_.alpha_chn_increase = this->declare_parameter<double>("alpha_channel_per_detected", 0.12);
    params_.IoU_treshhold = this->declare_parameter<double>("IoU_threshold", 0.5);
    params_.marker_color = this->declare_parameter<std::vector<double>>("marker_color", {0.0f, 1.0f, 0.0f});
    // clang-format on
}

void CalibrationNode::initSignals() {
    slot_.image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera_1/bgr/image",
        10,
        std::bind(&CalibrationNode::handleFrame, this, std::placeholders::_1));
    service_.button_service = this->create_service<std_srvs::srv::SetBool>(
        "/calibration/button",
        std::bind(
            &CalibrationNode::onButtonClick, this, std::placeholders::_1, std::placeholders::_2));

    signal_.chessboard_preview_pub =
        this->create_publisher<sensor_msgs::msg::Image>("/calibration/chessboard_preview_1", 10);
    signal_.detected_boards =
        this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>("/calibration/marker_1", 10);
    signal_.calibration_state =
        this->create_publisher<std_msgs::msg::Int16>("/calibration/state", 10);
    timer_.calibration_state_timer_ =
        this->create_wall_timer(1000ms, std::bind(&CalibrationNode::publishCalibrationState, this));
}

void CalibrationNode::handleFrame(const sensor_msgs::msg::Image::ConstPtr& msg) {
    if (state_.calibration_state != CAPTURING) {
        return;
    }

    cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    std::vector<cv::Point2f> detected_corners;
    bool found = cv::findChessboardCorners(
        image_ptr->image, params_.pattern_size, detected_corners, cv::CALIB_CB_FAST_CHECK);
    if (found) {
        bool req_to_add = checkMaxSimilarity(detected_corners);
        if (req_to_add) {
            if (params_.publish_chessboard_preview) {
                state_.markers_array.markers.push_back(
                    getMarkerFromCorners(detected_corners, image_ptr));
                signal_.detected_boards->publish(state_.markers_array);
            }

            state_.detected_corners_all.push_back(detected_corners);
            state_.object_points_all.push_back(params_.square_obj_points);
            if (params_.auto_calibrate &&
                state_.detected_corners_all.size() >= MIN_FRAMES_FOR_CALIBRATION) {
                calibrate();
            }
        }
    }
    if (params_.publish_chessboard_preview) {
        cv::drawChessboardCorners(image_ptr->image, params_.pattern_size, detected_corners, found);
        signal_.chessboard_preview_pub->publish(*image_ptr->toImageMsg());
    }
}

void CalibrationNode::onButtonClick(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    if (state_.calibration_state == CAPTURING) {
        calibrate();
    } else if (state_.calibration_state == NOT_CALIBRATED) {
        state_.calibration_state = CAPTURING;
    } else if (state_.calibration_state == OK_CALIBRATION) {
        state_.calibration_state = CAPTURING;
        state_.detected_corners_all.clear();
        state_.object_points_all.clear();
        state_.markers_array.markers.clear();
    }
}

bool CalibrationNode::checkMaxSimilarity(std::vector<cv::Point2f> corners) const {
    Polygon new_poly, prev_poly;
    const std::vector<Point> new_points = getPoints(corners, params_.pattern_size);
    boost::geometry::append(new_poly, new_points);

    double max_value = 0.0;

    for (const std::vector<cv::Point2f> prev_corners : state_.detected_corners_all) {
        const std::vector<Point> prev_points = getPoints(prev_corners, params_.pattern_size);
        boost::geometry::append(prev_poly, prev_points);

        std::deque<Polygon> union_poly;
        boost::geometry::union_(prev_poly, new_poly, union_poly);

        std::deque<Polygon> inter_poly;
        boost::geometry::intersection(prev_poly, new_poly, inter_poly);

        double current_value = 0.0;

        if (!inter_poly.empty()) {
            current_value = boost::geometry::area(inter_poly.front()) /
                            boost::geometry::area(union_poly.front());
        }

        boost::geometry::clear(prev_poly);
        max_value = std::max(max_value, current_value);
        if (max_value > params_.IoU_treshhold) {
            return false;
        }
    }

    return max_value <= params_.IoU_treshhold;
}

void CalibrationNode::publishCalibrationState() const {
    std_msgs::msg::Int16 msg;
    msg.data = state_.calibration_state;
    signal_.calibration_state->publish(msg);
}

visualization_msgs::msg::ImageMarker CalibrationNode::getMarkerFromCorners(
    std::vector<cv::Point2f>& detected_corners, cv_bridge::CvImagePtr image_ptr) {
    visualization_msgs::msg::ImageMarker marker;

    marker.header.frame_id = image_ptr->header.frame_id;
    marker.header.stamp = image_ptr->header.stamp;
    marker.ns = "calibration";
    marker.id = ++state_.last_marker_id;
    marker.type = visualization_msgs::msg::ImageMarker::POLYGON;
    marker.action = visualization_msgs::msg::ImageMarker::ADD;

    marker.position.x = 0;
    marker.position.y = 0;

    marker.scale = 1.;

    marker.filled = 1;
    marker.fill_color.r = params_.marker_color[0];
    marker.fill_color.g = params_.marker_color[1];
    marker.fill_color.b = params_.marker_color[2];
    marker.fill_color.a = params_.alpha_chn_increase;

    cv::Point2f board_cv_corners[4] = {
        detected_corners[0],
        detected_corners[params_.pattern_size.width - 1],
        detected_corners[params_.pattern_size.width * params_.pattern_size.height - 1],
        detected_corners[params_.pattern_size.width * (params_.pattern_size.height - 1)],
    };

    std::vector<geometry_msgs::msg::Point> marker_points(4);
    for (int i = 0; i < 4; ++i) {
        marker_points[i] = initPoint(board_cv_corners[i]);
    }
    marker.points = marker_points;

    return marker;
}

void CalibrationNode::calibrate() {
    state_.calibration_state = CALIBRATING;
    publishCalibrationState();
    RCLCPP_INFO_STREAM(this->get_logger(), "Calibration inialized");
    double rms = cv::calibrateCamera(
        state_.object_points_all,
        state_.detected_corners_all,
        params_.frame_size_,
        intrinsic_params_.camera_matrix,
        intrinsic_params_.dist_coefs,
        intrinsic_params_.rotation_vectors,
        intrinsic_params_.translation_vectors);
    RCLCPP_INFO_STREAM(this->get_logger(), "Calibration done with error of " << rms);

    if (rms < params_.min_accepted_error) {
        saveCalibParams();
        state_.calibration_state = OK_CALIBRATION;
    } else {
        RCLCPP_INFO_STREAM(this->get_logger(), "Continue taking frames");

        state_.calibration_state = BAD_CALIBRATION;
        publishCalibrationState();
        rclcpp::sleep_for(3s);
        state_.calibration_state = CAPTURING;
    }
    publishCalibrationState();
}

void CalibrationNode::saveCalibParams() const {
    const std::string path_to_yaml_file = params_.path_to_save_params;
    const bool status = intrinsic_params_.save(path_to_yaml_file);
    if (!status) {
        RCLCPP_ERROR_STREAM(
            this->get_logger(), "Error when writing params to " << path_to_yaml_file);
        abort();
    }
    RCLCPP_INFO_STREAM(
        this->get_logger(), "Calibration result saved successfully to " << path_to_yaml_file);
}
}  // namespace handy::calibration