#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/core/core.hpp>

#include "CameraApi.h"
#include "params.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <algorithm>

namespace handy::camera {

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();
    static void cameraCallback(
        CameraHandle idx, BYTE* raw_buffer, tSdkFrameHead* frame_info, PVOID camera_node_instance);

  private:
    void applyCameraParameters();
    void applyParamsToCamera(int camera_idx);
    int handleToId(int camera_handle);

    void triggerOnTimer();
    void handleFrame(CameraHandle idx, BYTE* raw_buffer, tSdkFrameHead* frame_info);

    void publishRawImage(uint8_t* buffer, const rclcpp::Time& timestamp, int camera_idx);
    void publishBGRImage(uint8_t* buffer, const rclcpp::Time& timestamp, int camera_idx);

    void abortIfNot(std::string_view msg, int status);
    void abortIfNot(std::string_view msg, int camera_idx, int status);

    struct Params {
        cv::Size preview_frame_size = cv::Size(640, 480);
        std::chrono::duration<double> latency{50.0};
        std::string calibration_file_path = "";
        int camera_num = 0;
        bool publish_bgr = false;
        bool publish_bgr_preview = false;
        bool publish_raw = false;
        bool publish_raw_preview = false;
        bool publish_rectified_preview = false;
        bool hardware_triger = false;
        int max_buffer_size = 0;
    } param_{};

    struct State {
        rclcpp::Time last_trigger_time;
        std::vector<int> frame_cnts;
    } state_{};

    std::vector<int> camera_handles_ = {};
    std::map<int, int> camera_idxs = {};
    std::unique_ptr<uint8_t[]> bgr_buffer_ = nullptr;
    std::vector<tSdkFrameHead> frame_info_ = {};
    std::vector<CameraIntrinsicParameters> cameras_intrinsics_ = {};
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

    struct CallbackGroups {
        rclcpp::CallbackGroup::SharedPtr trigger_timer = nullptr;
    } call_group_{};

    struct Timers {
        rclcpp::TimerBase::SharedPtr camera_soft_trigger = nullptr;
    } timer_{};
};

}  // namespace handy::camera
