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

    aruco_detector_ = std::make_unique<cv::aruco::ArucoDetector>(dictionary, detector_params);

    state_.is_calibrated.resize(param_.camera_num);
    state_.marker_corners_all.resize(param_.camera_num);
    state_.marker_ids_all.resize(param_.camera_num);
    state_.polygons_all.resize(param_.camera_num);

    for (int i = 0; i < param_.camera_num; ++i) {
        state_.intrinsic_params.push_back(
            CameraIntrinsicParameters::loadFromYaml(param_.path_to_params, param_.camera_ids[i]));
    }
}

void CalibrationNode::declareLaunchParams(const YAML::Node& param_node) {
    param_.path_to_params = param_node["calibration_file_path"].as<std::string>();
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

    aruco_detector_->detectMarkers(image, current_corners, current_ids);

    if (current_corners.size() < 20) {
        return false;
    }
    param_.aruco_board.matchImagePoints(
        current_corners, current_ids, current_obj_points, current_image_points);

    if (current_ids.size() < 30) {
        return false;
    }

    if (!checkMaxSimilarity(current_corners, camera_idx)) {
        return false;
    }

    state_.marker_corners_all[camera_idx].push_back(current_corners);
    state_.marker_ids_all[camera_idx].push_back(current_ids);

    state_.polygons_all[camera_idx].push_back(convertToBoostPolygon(current_corners));
    return true;
}

bool CalibrationNode::checkMaxSimilarity(
    std::vector<std::vector<cv::Point2f>>& corners, int camera_idx) const {
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
        state_.marker_corners_all[i].clear();
        state_.marker_ids_all[i].clear();
        state_.polygons_all[i].clear();
    }
}

void CalibrationNode::clearLastDetection(int camera_idx) {
    state_.marker_corners_all[camera_idx].pop_back();
    state_.marker_ids_all[camera_idx].pop_back();
    state_.polygons_all[camera_idx].pop_back();
}

void CalibrationNode::fillImageObjectPoints(
    std::vector<std::vector<cv::Point2f>>& image_points,
    std::vector<std::vector<cv::Point3f>>& obj_points, int camera_idx) {
    for (int i = 0; i < state_.marker_corners_all[camera_idx].size(); ++i) {
        image_points.emplace_back();
        obj_points.emplace_back();
        param_.aruco_board.matchImagePoints(
            state_.marker_corners_all[camera_idx][i],
            state_.marker_ids_all[camera_idx][i],
            obj_points.back(),
            image_points.back());
    }
}

bool CalibrationNode::isMonoCalibrated(int camera_idx) { return state_.is_calibrated[camera_idx]; }

void CalibrationNode::calibrate(int camera_idx) {
    printf("Calibration inialized\n");
    int coverage_percentage = getImageCoverage(camera_idx);
    if (coverage_percentage < param_.required_board_coverage) {
        printf("Coverage of %d is not sufficient\n", coverage_percentage);
        return;
    }
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> obj_points;
    fillImageObjectPoints(image_points, obj_points, camera_idx);

    state_.intrinsic_params[camera_idx].image_size = *state_.frame_size;
    std::vector<double> per_view_errors;
    double rms = cv::calibrateCamera(
        obj_points,
        image_points,
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

    printf("Calibration done with error of %f and coverage of %d\n", rms, coverage_percentage);

    if (rms < param_.min_accepted_error) {
        state_.intrinsic_params[camera_idx].storeYaml(param_.path_to_params);
    }
}

void CalibrationNode::stereoCalibrate() {
    std::vector<std::vector<std::vector<cv::Point2f>>> image_points_all;
    std::vector<std::vector<std::vector<cv::Point3f>>> obj_points_all;
    for (int i = 0; i < param_.camera_num; ++i) {
        image_points_all.emplace_back();
        obj_points_all.emplace_back();
        fillImageObjectPoints(image_points_all[i], obj_points_all[i], i);
    }

    std::vector<std::vector<std::vector<cv::Point2f>>> image_points_common(param_.camera_num);
    std::vector<std::vector<cv::Point3f>> obj_points_common;
    for (size_t detection_idx = 0; detection_idx < image_points_all[0].size(); ++detection_idx) {
        if (obj_points_common.empty() || !obj_points_common.back().empty()) {
            obj_points_common.emplace_back();
            for (size_t camera_idx = 0; camera_idx < param_.camera_num; ++camera_idx) {
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
        state_.intrinsic_params[0].camera_matrix,
        state_.intrinsic_params[0].dist_coefs,
        state_.intrinsic_params[1].camera_matrix,
        state_.intrinsic_params[1].dist_coefs,
        *state_.frame_size,
        rotation,
        translation,
        cv::noArray(),
        cv::noArray(),
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON));
    cv::Mat rotation_vector;
    cv::Rodrigues(rotation, rotation_vector);

    printf("Calibration done with error of %f \n", rms);

    cv::Mat zero_transformation = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    if (!CameraIntrinsicParameters::saveStereoCalibration(
            param_.path_to_params,
            zero_transformation,
            zero_transformation,
            image_points_common[0],
            1)) {
        printf(
            "Failed to save result of stereo calibration (id=%d) to the file: %s\n",
            1,
            param_.path_to_params.c_str());
        std::exit(EXIT_FAILURE);
    }

    if (!CameraIntrinsicParameters::saveStereoCalibration(
            param_.path_to_params, rotation_vector, translation, image_points_common[1], 2)) {
        printf(
            "Failed to save result of stereo calibration (id=%d) to the file: %s\n",
            2,
            param_.path_to_params.c_str());
        std::exit(EXIT_FAILURE);
    }
    printf("Saved stereo calibration result to %s\n", param_.path_to_params.c_str());
}
}  // namespace handy::calibration
