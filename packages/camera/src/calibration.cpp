#include "calibration.h"
#include "params.h"

#include <deque>
#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

/**
 * Converts Boost point to ROS2 message
 *
 * @param boost_point Boost point that will be converted
 * @return ROS2 Point message
 */
geometry_msgs::msg::Point toMsgPoint(const handy::calibration::Point& boost_point) {
    geometry_msgs::msg::Point point;
    point.x = boost_point.x();
    point.y = boost_point.y();
    return point;
}

}  // namespace
namespace handy::calibration {
/**
 * Converts all corners of detected chessboard into a convex hull
 *
 * @param corners vector of OpenCV points that should be covered by a convex hull
 * @return convex hull as a boost polygon
 */
Polygon convertToBoostPolygon(const std::vector<cv::Point2f>& corners) {
    Polygon poly;
    Polygon hull;
    for (size_t i = 0; i < corners.size(); ++i) {
        poly.outer().emplace_back(corners[i].x, corners[i].y);
    }
    boost::geometry::convex_hull(poly, hull);
    return hull;
}

/**
 * Converts all detected charuco corners into a convex hull polyline
 *
 * @param detected_corners vector of OpenCV points that should be covered by a convex hull
 * @return vector of markers to display
 */
std::vector<geometry_msgs::msg::Point> toBoardMsgCorners(
    const std::vector<cv::Point2f>& detected_corners) {
    Polygon poly;
    Polygon hull;
    for (size_t i = 0; i < detected_corners.size(); ++i) {
        poly.outer().emplace_back(detected_corners[i].x, detected_corners[i].y);
    }
    boost::geometry::convex_hull(poly, hull);

    std::vector<geometry_msgs::msg::Point> marker_points;
    for (size_t i = 0; i < hull.outer().size(); ++i) {
        marker_points.push_back(toMsgPoint(hull.outer()[i]));
    }
    return marker_points;
}

/**
 * Calculates intersection over union metric as a measure of similarity between two polygons
 *
 * @param first Boost polygon
 * @param second Boost polygon
 * @return the metric value
 */

double getIou(const Polygon& first, const Polygon& second) {
    std::deque<Polygon> inter_poly;
    boost::geometry::intersection(first, second, inter_poly);

    double iou = 0.0;
    if (inter_poly.size() > 1) {
        throw std::logic_error("Intersection is not valid");
    }
    if (!inter_poly.empty()) {
        double inter_area = boost::geometry::area(inter_poly.front());
        iou = inter_area /
              (boost::geometry::area(first) + boost::geometry::area(second) - inter_area);
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

    initSignals();
    RCLCPP_INFO_STREAM(this->get_logger(), "Signals initialised");

    for (int i = 0; i < kPatternSize.height; ++i) {
        for (int j = 0; j < kPatternSize.width; ++j) {
            param_.square_obj_points.emplace_back(i, j, 0);
        }
    }

    for (size_t i = 0; i < param_.cameras_to_calibrate.size(); ++i) {
        // TODO add loading params from YAML, PR #24
        // intrinsics_.push_back(CameraIntrinsicParameters::loadFromYaml(
        //     param_.path_to_params, param_.cameras_to_calibrate[i]));
        // state_.is_mono_calibrated.push_back(intrinsics_.back().isCalibrated());
        state_.is_mono_calibrated.push_back(false);
        intrinsics_.emplace_back();
        state_.board_markers_array.emplace_back();
        state_.board_corners_array.emplace_back();
        state_.detected_ids_all.emplace_back();
        state_.detectected_corners_all.emplace_back();
        state_.polygons_all.emplace_back();
    }
    state_.global_calibration_state = kNotCalibrated;

    timer_.calibration_state = this->create_wall_timer(
        std::chrono::milliseconds(250), [this] { publishCalibrationState(); });

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    param_.charuco_board = cv::aruco::CharucoBoard(kPatternSize, 0.04f, 0.02f, dictionary);
    param_.charuco_board.setLegacyPattern(true);

    cv::aruco::DetectorParameters detector_params;
    detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    detector_params.cornerRefinementWinSize = 10;
    cv::aruco::CharucoParameters board_params;

    charuco_detector_ = std::make_unique<cv::aruco::CharucoDetector>(
        param_.charuco_board, board_params, detector_params);

    for (size_t i = 0; i < param_.cameras_to_calibrate.size(); ++i) {
        signal_.detected_boards[i]->publish(state_.board_markers_array[i]);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "Init completed");
}

void CalibrationNode::declareLaunchParams() {
    // clang-format off
    param_.publish_preview_markers = this->declare_parameter<bool>("publish_preview_markers", true);
    param_.path_to_params = this->declare_parameter<std::string>("calibration_file_path", "param_path");

    param_.required_board_coverage = this->declare_parameter<double>("required_board_coverage", 0.7);
    param_.min_accepted_error = this->declare_parameter<double>("min_accepted_calib_error", 0.75);
    param_.iou_threshold = this->declare_parameter<double>("iou_threshold", 0.5);
    param_.marker_color=this->declare_parameter<std::vector<double>>("marker_color", {0.0f, 1.0f, 0.0f, 0.12f});

    param_.cameras_to_calibrate = this->declare_parameter<std::vector<int64_t>>("cameras_to_calibrate", {1, 2});
    param_.fps = this->declare_parameter<int64_t>("fps", 20);
    RCLCPP_INFO(this->get_logger(), "fps: %ld", param_.fps);
    // clang-format on

    // no more than 2 camera are supported at the moment
    if (param_.cameras_to_calibrate.empty() || param_.cameras_to_calibrate.size() > 2) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Provided %ld cameras instead of 1 or 2",
            param_.cameras_to_calibrate.size());
    }

    for (size_t i = 0; i < param_.cameras_to_calibrate.size(); i++) {
        param_.id_to_idx[param_.cameras_to_calibrate[i]] = i;
    }
}

void CalibrationNode::initSignals() {
    call_group_.handle_frame = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions options;
    options.callback_group = call_group_.handle_frame;
    for (size_t i = 0; i < param_.cameras_to_calibrate.size(); ++i) {
        const std::string calib_name_base =
            "/camera_" + std::to_string(param_.cameras_to_calibrate[i]);
        auto callback = [this, i](const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
            handleFrame(msg, i);
        };
        slot_.image_sub.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
            calib_name_base + "/bgr/image", 10, callback, options));

        signal_.detected_boards.push_back(
            this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>(
                calib_name_base + "/board_markers", 10));
        signal_.detected_corners.push_back(
            this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>(
                calib_name_base + "/corner_markers", 10));
    }

    service_.button_service = this->create_service<camera_srvs::srv::CalibrationCommand>(
        "/calibration/button",
        std::bind(
            &CalibrationNode::onButtonClick, this, std::placeholders::_1, std::placeholders::_2));

    signal_.calibration_state =
        this->create_publisher<std_msgs::msg::Int16>("/calibration/state", 10);
}

void CalibrationNode::handleFrame(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg, size_t camera_idx) {
    // exit callback if frames should not be captured or if camera_idx is already calibrated
    if ((state_.global_calibration_state != kCapturing &&
         state_.global_calibration_state != kStereoCapturing) ||
        (state_.is_mono_calibrated[camera_idx] && state_.global_calibration_state == kCapturing)) {
        return;
    }

    cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    if (!state_.frame_size) {
        state_.frame_size =
            std::make_optional<cv::Size>(image_ptr->image.cols, image_ptr->image.rows);
    }
    if (state_.frame_size->width != image_ptr->image.cols ||
        state_.frame_size->height != image_ptr->image.rows) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Camera ID=%ld provided (%d, %d), expected (%d, %d)",
            param_.cameras_to_calibrate[camera_idx],
            image_ptr->image.cols,
            image_ptr->image.rows,
            state_.frame_size->width,
            state_.frame_size->height);
    }

    std::vector<int> current_ids;
    std::vector<cv::Point2f> current_corners;
    std::vector<cv::Point2f> current_image_points;
    std::vector<cv::Point3f> current_obj_points;

    charuco_detector_->detectBoard(image_ptr->image, current_corners, current_ids);

    if (current_corners.size() < 20) {
        state_.board_corners_array[camera_idx].markers.clear();
        signal_.detected_corners[camera_idx]->publish(state_.board_corners_array[camera_idx]);
        return;
    }

    param_.charuco_board.matchImagePoints(
        current_corners, current_ids, current_obj_points, current_image_points);

    state_.board_corners_array[camera_idx].markers.clear();
    if (current_ids.size() < 30) {
        signal_.detected_corners[camera_idx]->publish(state_.board_corners_array[camera_idx]);
        return;
    }
    if (param_.publish_preview_markers) {
        appendCornerMarkers(current_corners, camera_idx);
        signal_.detected_corners[camera_idx]->publish(state_.board_corners_array[camera_idx]);
    }

    // if capturing frames for stereo calibration is in progress
    // then no need to check for similarity
    if (!checkMaxSimilarity(current_corners, camera_idx)) {
        return;
    }

    if (param_.publish_preview_markers) {
        state_.board_markers_array[camera_idx].markers.push_back(
            getBoardMarkerFromCorners(current_corners, image_ptr->header));
        signal_.detected_boards[camera_idx]->publish(state_.board_markers_array[camera_idx]);
    }

    std::unique_ptr<int[]> new_ids(new int[current_ids.size()]);
    std::copy(current_ids.begin(), current_ids.end(), new_ids.get());
    // timestamp is stored in millisecond
    const size_t frame_timestamp_ms =
        1000ull * image_ptr->header.stamp.sec + image_ptr->header.stamp.nanosec / 1000000ull;
    state_.detected_ids_all[camera_idx][frame_timestamp_ms] =
        std::make_pair(current_ids.size(), std::move(new_ids));

    std::unique_ptr<cv::Point2f[]> new_corners(new cv::Point2f[current_corners.size()]);
    std::copy(current_corners.begin(), current_corners.end(), new_corners.get());
    state_.detectected_corners_all[camera_idx][frame_timestamp_ms] =
        std::make_pair(current_corners.size(), std::move(new_corners));

    state_.polygons_all[camera_idx].push_back(convertToBoostPolygon(current_corners));
}

void CalibrationNode::onButtonClick(
    const camera_srvs::srv::CalibrationCommand::Request::SharedPtr& request,
    const camera_srvs::srv::CalibrationCommand::Response::SharedPtr& /*response*/) {
    if (state_.global_calibration_state == kStereoCalibrating ||
        state_.global_calibration_state == kMonoCalibrating) {
        return;
    }
    if (request->cmd == kStart) {
        state_.global_calibration_state = kCapturing;
        RCLCPP_INFO_STREAM(this->get_logger(), "State set to capturing");

    } else if (request->cmd == kCalibrate) {
        if (state_.global_calibration_state == kStereoCapturing) {
            stereoCalibrate();
            return;
        }
        for (size_t i = 0; i < param_.cameras_to_calibrate.size(); ++i) {
            if (state_.is_mono_calibrated[i]) {
                continue;
            }
            calibrate(i);
        }
    } else if (request->cmd == kReset) {
        state_.global_calibration_state = kNotCalibrated;
        handleResetCommand();
    }
}

bool CalibrationNode::checkMaxSimilarity(
    std::vector<cv::Point2f>& corners, size_t camera_idx) const {
    Polygon new_poly = convertToBoostPolygon(corners);
    return std::all_of(
        state_.polygons_all[camera_idx].begin(),
        state_.polygons_all[camera_idx].end(),
        [&](const Polygon& prev_poly) {
            if (state_.global_calibration_state ==
                kStereoCapturing) {  // we should take more images to sync stereopairs
                return getIou(prev_poly, new_poly) <= param_.iou_threshold + 0.2;
            }
            return getIou(prev_poly, new_poly) <= param_.iou_threshold;
        });
}

int CalibrationNode::getImageCoverage(size_t camera_idx) const {
    MultiPolygon current_coverage;
    for (const Polygon& new_poly : state_.polygons_all[camera_idx]) {
        MultiPolygon tmp_poly;
        boost::geometry::union_(current_coverage, new_poly, tmp_poly);
        current_coverage = tmp_poly;
    }
    float ratio = boost::geometry::area(current_coverage) /
                  (state_.frame_size->height * state_.frame_size->width);
    return static_cast<int>(ratio * 100);
}

void CalibrationNode::publishCalibrationState() const {
    std_msgs::msg::Int16 msg;
    msg.data = state_.global_calibration_state;
    signal_.calibration_state->publish(msg);
}

visualization_msgs::msg::ImageMarker CalibrationNode::getBoardMarkerFromCorners(
    std::vector<cv::Point2f>& detected_corners, std_msgs::msg::Header& header) {
    visualization_msgs::msg::ImageMarker marker;

    marker.header.frame_id = header.frame_id;
    marker.header.stamp = header.stamp;
    marker.ns = "calibration";
    marker.id = state_.last_marker_id.fetch_add(1);
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

    marker.points = toBoardMsgCorners(detected_corners);
    return marker;
}

visualization_msgs::msg::ImageMarker CalibrationNode::getCornerMarker(cv::Point2f point) {
    visualization_msgs::msg::ImageMarker marker;

    marker.ns = "calibration";
    marker.id = state_.last_marker_id.fetch_add(1);
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

void CalibrationNode::appendCornerMarkers(
    const std::vector<cv::Point2f>& detected_corners, size_t camera_idx) {
    for (size_t i = 0; i < detected_corners.size(); ++i) {
        visualization_msgs::msg::ImageMarker marker = getCornerMarker(detected_corners[i]);
        state_.board_corners_array[camera_idx].markers.push_back(marker);
    }
}

void CalibrationNode::handleBadCalibration(size_t camera_idx) {
    RCLCPP_INFO(this->get_logger(), "Continue taking frames for ID=%ld", camera_idx);

    state_.global_calibration_state = kCapturing;
}

void CalibrationNode::handleResetCommand(int camera_idx) {
    for (size_t i = 0; i < param_.cameras_to_calibrate.size(); ++i) {
        if (camera_idx != -1 && static_cast<size_t>(camera_idx) != i) {
            continue;
        }
        state_.detected_ids_all[i].clear();
        state_.detectected_corners_all[i].clear();
        state_.board_markers_array[i].markers.clear();
        state_.polygons_all[i].clear();
        signal_.detected_boards[i]->publish(state_.board_markers_array[i]);  // to clear the screen
    }
}

void CalibrationNode::fillImageObjectPoints(
    std::vector<std::vector<cv::Point2f>>& image_points,
    std::vector<std::vector<cv::Point3f>>& obj_points, int camera_idx) {
    for (auto id_iter = state_.detectected_corners_all[camera_idx].begin();
         id_iter != state_.detectected_corners_all[camera_idx].end();
         ++id_iter) {
        const size_t timestamp = id_iter->first;
        auto& [size, data_ptr] = id_iter->second;

        // CV_32FC2 -- 4 bytes, 2 channels -- x and y coordinates
        cv::Mat current_corners(cv::Size{1, static_cast<int>(size)}, CV_32FC2, data_ptr.get());
        cv::Mat current_ids(
            cv::Size{1, static_cast<int>(size)},
            CV_32SC1,
            state_.detected_ids_all[camera_idx][timestamp].second.get());

        image_points.emplace_back();
        obj_points.emplace_back();
        param_.charuco_board.matchImagePoints(
            current_corners, current_ids, obj_points.back(), image_points.back());
    }
}

void CalibrationNode::calibrate(size_t camera_idx) {
    state_.global_calibration_state = kMonoCalibrating;
    publishCalibrationState();
    RCLCPP_INFO(this->get_logger(), "Calibration ID=%ld inialized", camera_idx);

    std::vector<std::vector<cv::Point2f>> image_points{};
    std::vector<std::vector<cv::Point3f>> obj_points{};

    // fill image and object points
    fillImageObjectPoints(image_points, obj_points, camera_idx);

    int coverage_percentage = getImageCoverage(camera_idx);
    if (coverage_percentage < param_.required_board_coverage) {
        RCLCPP_INFO(this->get_logger(), "Coverage of %d is not sufficient", coverage_percentage);
        handleBadCalibration(camera_idx);
        return;
    }

    intrinsics_[camera_idx].image_size = *state_.frame_size;
    std::vector<double> per_view_errors;

    double rms = cv::calibrateCamera(
        obj_points,
        image_points,
        *state_.frame_size,
        intrinsics_[camera_idx].camera_matrix,
        intrinsics_[camera_idx].dist_coefs,
        cv::noArray(),
        cv::noArray(),
        cv::noArray(),
        cv::noArray(),
        per_view_errors,
        cv::CALIB_FIX_S1_S2_S3_S4,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON));

    RCLCPP_INFO(
        this->get_logger(),
        "Calibration done with error of %f and coverage of %d",
        rms,
        coverage_percentage);

    // TODO save the intrinsic by ID accordingly (PR #24)
    if (rms < param_.min_accepted_error) {
        state_.is_mono_calibrated[camera_idx] = true;
        // intrinsics_[camera_idx].storeYaml(param_.path_to_params);
        handleResetCommand(camera_idx);
        const bool all_calibrated = std::all_of(
            state_.is_mono_calibrated.begin(), state_.is_mono_calibrated.end(), [](bool elem) {
                return elem;
            });
        if (all_calibrated) {
            state_.global_calibration_state = kStereoCapturing;
        }
    } else {
        handleBadCalibration(camera_idx);
    }
}

void CalibrationNode::stereoCalibrate() {
    state_.global_calibration_state = kStereoCalibrating;
    publishCalibrationState();
    RCLCPP_INFO(this->get_logger(), "Stereo calibration inialized");

    std::vector<std::vector<std::vector<cv::Point2f>>> image_points_all(
        param_.cameras_to_calibrate.size());
    std::vector<std::vector<std::vector<cv::Point3f>>> obj_points_all(
        param_.cameras_to_calibrate.size());

    size_t chosen_detections_cnt = 0;

    while (std::all_of(
        state_.detectected_corners_all.begin(),
        state_.detectected_corners_all.end(),
        [&](const auto& elem) { return !elem.empty() && chosen_detections_cnt < elem.size(); })) {
        size_t idx_of_min_timestamp = 0;
        size_t min_timestamp = std::numeric_limits<size_t>::max();
        size_t max_timestamp = 0;
        for (size_t i = 0; i < state_.detectected_corners_all.size(); ++i) {
            auto current_map_elem = state_.detectected_corners_all[i].begin();
            std::advance(current_map_elem, chosen_detections_cnt);
            size_t current_timestamp = current_map_elem->first;

            max_timestamp = std::max(max_timestamp, current_timestamp);
            if (current_timestamp < min_timestamp) {
                min_timestamp = current_timestamp;
                idx_of_min_timestamp = i;
            }
        }
        if (max_timestamp - min_timestamp <
            static_cast<size_t>(1000 / param_.fps)) {  // found simultaneous snapshot
            ++chosen_detections_cnt;
        } else {
            auto corners_iter_to_delete =
                state_.detectected_corners_all[idx_of_min_timestamp].begin();
            std::advance(corners_iter_to_delete, chosen_detections_cnt);
            state_.detectected_corners_all[idx_of_min_timestamp].erase(corners_iter_to_delete);

            auto ids_iter_to_delete = state_.detected_ids_all[idx_of_min_timestamp].begin();
            std::advance(ids_iter_to_delete, chosen_detections_cnt);
            state_.detected_ids_all[idx_of_min_timestamp].erase(ids_iter_to_delete);
        }
    }

    for (size_t camera_idx = 0; camera_idx < param_.cameras_to_calibrate.size(); ++camera_idx) {
        auto last_valid_corner_iter = state_.detectected_corners_all[camera_idx].begin();
        std::advance(last_valid_corner_iter, chosen_detections_cnt);
        state_.detectected_corners_all[camera_idx].erase(
            last_valid_corner_iter, state_.detectected_corners_all[camera_idx].end());

        auto last_valid_id_iter = state_.detected_ids_all[camera_idx].begin();
        std::advance(last_valid_id_iter, chosen_detections_cnt);
        state_.detected_ids_all[camera_idx].erase(
            last_valid_id_iter, state_.detected_ids_all[camera_idx].end());
    }

    for (size_t camera_idx = 0; camera_idx < param_.cameras_to_calibrate.size(); ++camera_idx) {
        // fill image and object points
        fillImageObjectPoints(
            image_points_all[camera_idx], obj_points_all[camera_idx], static_cast<int>(camera_idx));
    }

    // for now we assume that we have 2 cameras
    std::vector<std::vector<std::vector<cv::Point2f>>> image_points_common(
        param_.cameras_to_calibrate.size());
    std::vector<std::vector<cv::Point3f>> obj_points_common;
    for (size_t detection_idx = 0; detection_idx < image_points_all[0].size(); ++detection_idx) {
        if (obj_points_common.empty() || !obj_points_common.back().empty()) {
            obj_points_common.emplace_back();
            for (size_t camera_idx = 0; camera_idx < param_.cameras_to_calibrate.size();
                 ++camera_idx) {
                image_points_common[camera_idx].emplace_back();
            }
        }

        for (size_t i = 0; i < obj_points_all[0][detection_idx].size(); ++i) {
            std::vector<size_t> found_indexes;
            if (std::all_of(
                    obj_points_all.begin(),
                    obj_points_all.end(),
                    [&](const auto& camera_obj_points) {
                        auto found = std::find(
                            camera_obj_points[detection_idx].begin(),
                            camera_obj_points[detection_idx].end(),
                            obj_points_all[0][detection_idx][i]);
                        if (found != camera_obj_points[detection_idx].end()) {
                            found_indexes.push_back(
                                std::distance(camera_obj_points[detection_idx].begin(), found));
                            return true;
                        }
                        return false;
                    })) {
                obj_points_common.back().push_back(obj_points_all[0][detection_idx][i]);
                for (size_t camera_idx = 0; camera_idx < found_indexes.size(); ++camera_idx) {
                    const cv::Point2f& common_point_to_add =
                        image_points_all[camera_idx][detection_idx][found_indexes[camera_idx]];
                    image_points_common[camera_idx].back().push_back(common_point_to_add);
                }
            }
        }
    }

    cv::Mat rotation;
    cv::Mat translation;

    double rms = cv::stereoCalibrate(
        obj_points_common,
        image_points_common[0],
        image_points_common[1],
        intrinsics_[0].camera_matrix,
        intrinsics_[0].dist_coefs,
        intrinsics_[1].camera_matrix,
        intrinsics_[1].dist_coefs,
        *state_.frame_size,
        rotation,
        translation,
        cv::noArray(),
        cv::noArray(),
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON));

    RCLCPP_INFO(this->get_logger(), "Calibration done with error of %f ", rms);

    if (rms < param_.min_accepted_error) {
        if (!CameraIntrinsicParameters::saveStereoCalibration(
                param_.path_to_params, rotation, translation)) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to save result of stereo calibration to the file: %s",
                param_.path_to_params.c_str());
            std::exit(EXIT_FAILURE);
        }
        state_.global_calibration_state = kOkCalibration;
    } else {
        RCLCPP_INFO(this->get_logger(), "Continue taking frames for stereo calibration");
        state_.global_calibration_state = kStereoCapturing;
    }
}
}  // namespace handy::calibration
