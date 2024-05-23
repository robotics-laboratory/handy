#include "calibration.h"

#include <cstdlib>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

int main() {
    const YAML::Node param_node =
        YAML::LoadFile("../../share/camera/launch/calibration_launch.yaml")["parameters"];
    auto camera_num = param_node["camera_num"].as<int>();
    auto camera_ids = param_node["camera_ids"].as<std::vector<int>>();
    auto stereo_directories = param_node["stereo_source_dirs"].as<std::vector<std::string>>();

    handy::calibration::CalibrationNode calibration_node(param_node);

    // mono calibration
    for (int i = 0; i < camera_num; ++i) {
        if (!param_node["mono_source_dirs"][std::to_string(camera_ids[i])].IsDefined()) {
            continue;
        }
        const int current_camera_id = camera_ids[i];
        const YAML::Node source_dir_node =
            param_node["mono_source_dirs"][std::to_string(current_camera_id)];
        if (!source_dir_node.IsDefined()) {
            continue;
        }

        const auto mono_source_dir = source_dir_node.as<std::string>();
        int detected_boards_counter = 0;
        int total = 0;
        for (const auto& entry : std::filesystem::directory_iterator(mono_source_dir)) {
            ++total;
            printf("reading %d\n", total);
            cv::Mat raw_image = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
            if (raw_image.empty()) {
                printf(
                    "error while reading %s for camera idx=%d\n",
                    entry.path().filename().c_str(),
                    i);
                exit(EXIT_FAILURE);
            }
            cv::Mat image;
            cv::cvtColor(raw_image, image, cv::COLOR_BayerBG2BGR);
            if (calibration_node.handleFrame(image, i)) {
                ++detected_boards_counter;
                printf(
                    "detected board on %s\ntotal is %d/%d\n",
                    entry.path().filename().c_str(),
                    detected_boards_counter,
                    total);
            }
        }
        calibration_node.calibrate(i);
    }
    calibration_node.clearDetections();
    printf("mono calibration done or loaded. Starting stereo calibration\n");

    // stereo calibration
    std::vector<int> detections_counters(camera_num, 0);
    for (const auto& entry : std::filesystem::directory_iterator(stereo_directories[0])) {
        const std::string base_filename = entry.path().filename();
        for (int i = 0; i < camera_num; ++i) {
            cv::Mat raw_image =
                cv::imread(stereo_directories[i] + base_filename, cv::IMREAD_GRAYSCALE);
            if (raw_image.empty()) {
                printf("error while reading %s for camera idx=%d\n", base_filename.c_str(), i);
                exit(EXIT_FAILURE);
            }
            cv::Mat image;
            cv::cvtColor(raw_image, image, cv::COLOR_BayerBG2BGR);

            if (!calibration_node.handleFrame(image, i)) {
                // delete the last detections across all camera in case one failed to detect a board
                for (int j = 0; j < i; ++j) {
                    calibration_node.clearLastDetection(j);
                    --detections_counters[j];
                }
                break;
            }
            ++detections_counters[i];
        }
    }
    printf("%d %d boards were detected\n", detections_counters[0], detections_counters[1]);
    calibration_node.stereoCalibrate();

    return 0;
}
