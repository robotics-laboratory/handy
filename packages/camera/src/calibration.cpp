#include "calibration.h"

namespace handy::calibration {

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
    if (!this->declare_parameter<bool>("calibration_needed", false)) {
        rclcpp::shutdown();
    }

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
    signals_.image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera_1/bgr/image",
        10,
        std::bind(&CalibrationNode::handleFrame, this, std::placeholders::_1));
    signals_.button_service = this->create_service<std_srvs::srv::SetBool>(
        "/calibration/button",
        std::bind(
            &CalibrationNode::onButtonClick, this, std::placeholders::_1, std::placeholders::_2));

    signals_.chessboard_preview_pub =
        this->create_publisher<sensor_msgs::msg::Image>("/calibration/chessboard_preview_1", 10);
    signals_.detected_boards =
        this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>("/calibration/marker_1", 10);
    signals_.calibration_state =
        this->create_publisher<std_msgs::msg::Int16>("/calibration/state", 10);
    calibration_state_timer_ =
        this->create_wall_timer(1000ms, std::bind(&CalibrationNode::publishCalibrationState, this));
}

void CalibrationNode::handleFrame(const sensor_msgs::msg::Image::ConstPtr& msg) {
    if (state_.calibration_state != CAPTURING_STATE) {
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
                signals_.detected_boards->publish(state_.markers_array);
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
        signals_.chessboard_preview_pub->publish(*image_ptr->toImageMsg());
    }
}

void CalibrationNode::onButtonClick(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    if (state_.calibration_state == CAPTURING_STATE) {
        calibrate();
    } else if (state_.calibration_state == NOT_CALIBRATED_STATE) {
        state_.calibration_state = CAPTURING_STATE;
    } else if (state_.calibration_state == OK_CALIBRATION_STATE) {
        state_.calibration_state = CAPTURING_STATE;
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
    signals_.calibration_state->publish(msg);
}

geometry_msgs::msg::Point CalibrationNode::initPoint(cv::Point2f cv_point) const {
    geometry_msgs::msg::Point point;
    point.x = cv_point.x;
    point.y = cv_point.y;
    return point;
}

visualization_msgs::msg::ImageMarker CalibrationNode::getMarkerFromCorners(
    std::vector<cv::Point2f>& detected_corners, cv_bridge::CvImagePtr image_ptr) {
    visualization_msgs::msg::ImageMarker marker;

    marker.header.frame_id = image_ptr->header.frame_id;
    marker.header.stamp = this->get_clock()->now();
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

    marker.lifetime = rclcpp::Duration::Duration::from_seconds(1000);

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
    state_.calibration_state = CALIBRATING_STATE;
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
    CalibrationPrecision calib_results = calcCalibPrecision();
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Calibration done with summ error of: " << calib_results.summ_repr_error
                                                << " and max error of: "
                                                << calib_results.max_repr_error);

    if (rms < params_.min_accepted_error) {
        saveCalibParams();
        state_.calibration_state = OK_CALIBRATION_STATE;
    } else {
        RCLCPP_INFO_STREAM(this->get_logger(), "Continue taking frames");

        state_.calibration_state = BAD_CALIBRATION_STATE;
        publishCalibrationState();
        rclcpp::sleep_for(3s);
        state_.calibration_state = CAPTURING_STATE;
    }
    publishCalibrationState();
}

CalibrationPrecision CalibrationNode::calcCalibPrecision() const {
    std::vector<std::vector<cv::Point2f>> repr_image_point;
    double repr_error = 0.0;
    double max_repr_error = 0.0;
    for (int i = 0; i < state_.detected_corners_all.size(); ++i) {
        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(
            state_.object_points_all[i],
            intrinsic_params_.rotation_vectors[i],
            intrinsic_params_.translation_vectors[i],
            intrinsic_params_.camera_matrix,
            intrinsic_params_.dist_coefs,
            projected_points);

        double current_error = cv::norm(state_.detected_corners_all[i], projected_points);
        repr_error += current_error;
        max_repr_error = std::max(max_repr_error, current_error);
    }
    CalibrationPrecision result{repr_error / state_.detected_corners_all.size(), max_repr_error};
    return result;
}

void CalibrationNode::saveCalibParams() const {
    std::string path_to_yaml_file = params_.path_to_save_params;
    std::ofstream param_file(path_to_yaml_file);
    if (!param_file) {
        RCLCPP_ERROR_STREAM(this->get_logger(), "invalid path " << path_to_yaml_file);
        abort();
    }

    YAML::Emitter output_yaml;
    output_yaml << YAML::BeginMap;

    output_yaml << YAML::Key << "camera_matrix";
    output_yaml << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            output_yaml << intrinsic_params_.camera_matrix(i, j);
        }
    }
    output_yaml << YAML::EndSeq;

    output_yaml << YAML::Key << "distorsion_coefs";
    output_yaml << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < 5; ++i) {
        output_yaml << intrinsic_params_.dist_coefs[i];
    }
    output_yaml << YAML::EndSeq;

    output_yaml << YAML::EndMap;

    param_file << output_yaml.c_str();
    param_file.close();

    RCLCPP_INFO_STREAM(
        this->get_logger(), "Calibration result saved successfully to " << path_to_yaml_file);
}
}  // namespace handy::calibration