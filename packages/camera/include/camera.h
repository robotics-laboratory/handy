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
#include <queue>
#include <mutex>

namespace handy::camera {

struct StampedImagePtr {
    std::shared_ptr<uint8_t[]> buffer = nullptr;
    tSdkFrameHead frame_info;
    rclcpp::Time timestamp;
    int source_idx = 0;

    StampedImagePtr(
        std::shared_ptr<uint8_t[]> buf, tSdkFrameHead* frame_inf, rclcpp::Time stamp, int source)
        : buffer(buf), frame_info(std::move(*frame_inf)), timestamp(stamp), source_idx(source) {}
};

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();
    
    static constexpr int NUM_OF_BUFFERS = 3;

  private:
    void applyCameraParameters();
    void applyParamsToCamera(int camera_idx);
    int handleToId(int camera_handle);

    void triggerOnTimer();
    void handleFrame(CameraHandle idx, BYTE* raw_buffer, tSdkFrameHead* frame_info);
    void handleQueue();

    void publishRawImage(uint8_t* buffer, rclcpp::Time timestamp, int camera_idx);
    void publishBGRImage(
        uint8_t* buffer, rclcpp::Time timestamp, int camera_idx, tSdkFrameHead& frame_inf);

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
        std::vector<std::queue<StampedImagePtr>> recieved_buffers;
        std::vector<std::mutex> recieved_buffers_mutexes;
        std::vector<std::vector<std::shared_ptr<uint8_t[]>>> free_buffers;
        std::vector<std::mutex> free_buffers_mutexes;
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
        rclcpp::TimerBase::SharedPtr camera_handle_queue_timer = nullptr;
    } timer_{};
};

}  // namespace handy::camera
