#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "CameraApi.h"
#include "params.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace handy::camera {

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();

  private:
    void applyCameraParameters();
    void applyParamsToCamera(int camera_idx);
    void initCalibParams();

    void handleOnTimer();

    void publishRawImage(uint8_t* buffer, rclcpp::Time timestamp, int camera_idx);
    void publishBGRImage(uint8_t* buffer, rclcpp::Time timestamp, int camera_idx);

    void abortIfNot(std::string_view msg, int status);
    void abortIfNot(std::string_view msg, int camera_idx, int status);
    
    int getMaxBufferSize();

    struct Params {
        cv::Size preview_frame_size = cv::Size(640, 480);
        std::chrono::duration<double> latency{50.0};
        std::vector<std::string> calibration_file_paths = {"param_save/camera_params.yaml"};
        int camera_num = 0;
        bool publish_bgr = false;
        bool publish_bgr_preview = false;
        bool publish_raw = false;
        bool publish_raw_preview = false;
        bool publish_rectified_preview = false;
        int max_buffer_size = 0;
    } param_{};

    std::vector<int> camera_handles_ = {};
    std::vector<uint8_t*> raw_buffer_ptr_ = {};
    std::unique_ptr<uint8_t[]> bgr_buffer_ = nullptr;
    std::vector<tSdkFrameHead> frame_info_ = {};
    std::vector<CameraUndistortModule> cameras_params_modules_ = {};
    std::vector<cv::Size> frame_sizes_ = {};

    struct Signals {
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> raw_img;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> bgr_img;

        // clang-format off
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> rectified_preview_img;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> raw_preview_img;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> bgr_preview_img;
        // clang-format on
    } signals_{};

    struct Timers {
        rclcpp::TimerBase::SharedPtr camera_capture = nullptr;
    } timer_{};
};

}  // namespace handy::camera
