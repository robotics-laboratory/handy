#include "calibration.h"

#include <deque>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

namespace handy::calibration {
/**
 * Converts all corners of detected chessboard into a convex hull
 *
 * @param corners vector of OpenCV points that should be covered by a convex hull
 * @return convex hull as a boost polygon
 */
Polygon convertToBoostPolygon(const std::vector<std::vector<cv::Point2f>>& corners) {
    Polygon poly;
    Polygon hull;
    for (size_t i = 0; i < corners.size(); ++i) {
        poly.outer().emplace_back(corners[i][0].x, corners[i][0].y);
    }
    boost::geometry::convex_hull(poly, hull);
    return hull;
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

CalibrationNode::CalibrationNode(const YAML::Node& param_node) {
    printf("Init started\n");

    declareLaunchParams(param_node);
    printf("Launch params declared\n");

    cv::Size pattern_size{7, 10};
    for (int i = 0; i < pattern_size.height; ++i) {
        for (int j = 0; j < pattern_size.width; ++j) {
            param_.square_obj_points.emplace_back(i, j, 0);
        }
    }

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    param_.charuco_board = cv::aruco::CharucoBoard(pattern_size, 0.06f, 0.04f, dictionary);
    param_.charuco_board.setLegacyPattern(false);
    param_.aruco_board = cv::aruco::GridBoard(cv::Size{5, 7}, 0.06f, 0.03f, dictionary);

    cv::aruco::DetectorParameters detector_params;
    detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    detector_params.cornerRefinementWinSize = 10;
    cv::aruco::CharucoParameters board_params;

    charuco_detector_ = std::make_unique<cv::aruco::CharucoDetector>(
        param_.charuco_board, board_params, detector_params);

    aruco_detector_ = std::make_unique<cv::aruco::ArucoDetector>(dictionary);

    state_.is_calibrated.resize(param_.camera_num);
    state_.image_points_all.resize(param_.camera_num);
    state_.obj_points_all.resize(param_.camera_num);
    state_.polygons_all.resize(param_.camera_num);

    for (int i = 0; i < param_.camera_num; ++i) {
        state_.intrinsic_params.push_back(CameraIntrinsicParameters::loadFromYaml(
            param_.path_to_save_params, param_.camera_ids[i]));
        // state_.intrinsic_params.emplace_back();
        // state_.intrinsic_params.back().camera_id = param_.camera_ids[i];
    }
}

void CalibrationNode::declareLaunchParams(const YAML::Node& param_node) {
    param_.path_to_save_params = param_node["calibration_file_path"].as<std::string>();
    param_.required_board_coverage = param_node["required_board_coverage"].as<double>();
    param_.min_accepted_error = param_node["min_accepted_calib_error"].as<double>();
    param_.iou_threshold = param_node["iou_threshold"].as<double>();
    param_.min_accepted_error = param_node["min_accepted_calib_error"].as<double>();
    param_.marker_color = param_node["marker_color"].as<std::vector<double>>();
    param_.camera_ids = param_node["camera_ids"].as<std::vector<int>>();
    param_.camera_num = param_node["camera_num"].as<int>();
}

bool CalibrationNode::handleFrame(const cv::Mat& image, int camera_idx) {
    if (!state_.frame_size) {
        state_.frame_size = std::make_optional<cv::Size>(image.cols, image.rows);
    }
    assert(state_.frame_size->width == image.cols && state_.frame_size->height == image.rows);

    std::vector<int> current_ids;
    std::vector<std::vector<cv::Point2f>> current_corners;
    std::vector<cv::Point2f> current_image_points;
    std::vector<cv::Point3f> current_obj_points;

    // charuco_detector_->detectBoard(image, current_corners, current_ids);
    aruco_detector_->detectMarkers(image, current_corners, current_ids);

    if (current_corners.size() < 20) {
        return false;
    }
    // param_.charuco_board.matchImagePoints(
    //     current_corners, current_ids, current_obj_points, current_image_points);
    param_.aruco_board.matchImagePoints(
        current_corners, current_ids, current_obj_points, current_image_points);

    if (current_ids.size() < 30) {
        return false;
    }
    // if (param_.publish_preview_markers) {
    //     appendCornerMarkers(current_corners);
    //     signal_.detected_corners->publish(state_.board_corners_array);
    // }
    if (!checkMaxSimilarity(current_corners, camera_idx)) {
        return false;
    }
    // if (param_.publish_preview_markers) {
    //     state_.board_markers_array.markers.push_back(
    //         getBoardMarkerFromCorners(current_corners, image_ptr->header));
    //     signal_.detected_boards->publish(state_.board_markers_array);
    // }
    for (int i = 0; i < current_ids.size(); ++i) {
        printf("%d ", current_ids[i]);
    }
    printf("\n");

    // state_.obj_points_all[camera_idx].push_back(current_obj_points);
    // state_.image_points_all[camera_idx].push_back(current_image_points);
    state_.obj_points_all[camera_idx].emplace_back();
    state_.image_points_all[camera_idx].emplace_back();

    state_.polygons_all[camera_idx].push_back(convertToBoostPolygon(current_corners));
    return true;
}

bool CalibrationNode::checkMaxSimilarity(std::vector<std::vector<cv::Point2f>>& corners, int camera_idx) const {
    Polygon new_poly = convertToBoostPolygon(corners);
    return std::all_of(
        state_.polygons_all[camera_idx].begin(),
        state_.polygons_all[camera_idx].end(),
        [&](const Polygon& prev_poly) {
            return getIou(prev_poly, new_poly) <= param_.iou_threshold;
        });
}

int CalibrationNode::getImageCoverage(int camera_idx) const {
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

void CalibrationNode::clearDetections() {
    for (int i = 0; i < param_.camera_num; ++i) {
        state_.image_points_all[i].clear();
        state_.obj_points_all[i].clear();
        state_.polygons_all[i].clear();
    }
}

void CalibrationNode::clearLastDetection(int camera_idx) {
    state_.image_points_all[camera_idx].pop_back();
    state_.obj_points_all[camera_idx].pop_back();
    state_.polygons_all[camera_idx].pop_back();
}

void CalibrationNode::calibrate(int camera_idx) {
    printf("Calibration inialized");
    int coverage_percentage = getImageCoverage(camera_idx);
    if (coverage_percentage < param_.required_board_coverage) {
        printf("Coverage of %d is not sufficient", coverage_percentage);
        return;
    }

    state_.intrinsic_params[camera_idx].image_size = *state_.frame_size;
    std::vector<double> per_view_errors;
    double rms = cv::calibrateCamera(
        state_.obj_points_all[camera_idx],
        state_.image_points_all[camera_idx],
        *state_.frame_size,
        state_.intrinsic_params[camera_idx].camera_matrix,
        state_.intrinsic_params[camera_idx].dist_coefs,
        cv::noArray(),
        cv::noArray(),
        cv::noArray(),
        cv::noArray(),
        per_view_errors,
        cv::CALIB_FIX_S1_S2_S3_S4,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON));

    printf("Calibration done with error of %f and coverage of %d", rms, coverage_percentage);

    if (rms < param_.min_accepted_error) {
        state_.intrinsic_params[camera_idx].storeYaml(param_.path_to_save_params);
    }
}
}  // namespace handy::calibration
