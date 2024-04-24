#include "calibration.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

namespace {
std::string ljust(std::string line_to_format, int required_length, char fill_chr) {
    while (line_to_format.size() < required_length) {
        line_to_format = fill_chr + line_to_format;
    }
    return line_to_format;
}
}  // namespace

int main(int argc, char* argv[]) {
    const YAML::Node param_node = YAML::LoadFile("calibration.yaml")["parameters"];
    auto camera_num = param_node["camera_num"].as<int>();
    auto stereo_directories = param_node["stereo_source_dirs"].as<std::vector<std::string>>();

    handy::calibration::CalibrationNode node(param_node);

    // mono calibration...

    // ...

    // stereo calibration
    int counters[2] = {0, 0};
    for (const auto& entry : std::filesystem::directory_iterator(stereo_directories[0])) {
        const std::string base_filename = entry.path().filename();
        // printf("%s\n", entry.path().filename().c_str());
        for (int i = 0; i < camera_num; ++i) {
            cv::Mat raw_image =
                cv::imread(stereo_directories[i] + base_filename, cv::IMREAD_GRAYSCALE);
            if (raw_image.empty()) {
                printf("error while reading %s for camera idx=%d\n", base_filename.c_str(), i);
                exit(EXIT_FAILURE);
            }
            cv::Mat image;
            cv::cvtColor(raw_image, image, cv::COLOR_BayerBG2BGR);

            if (!node.handleFrame(image, i)) {
                // delete the last detections across all camera in case one failed to detect a board
                for (int j = 0; j < i; ++j) {
                    node.clearLastDetection(j);
                    --counters[j];
                }
                break;
            }
            ++counters[i];
            printf("%d %d boards were detected\n", counters[0], counters[1]);
        }
    }
    printf("%d %d boards were detected finally\n", counters[0], counters[1]);

    return 0;
}
