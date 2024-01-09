#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/core/core.hpp>
#include <boost/lockfree/queue.hpp>

#include "CameraApi.h"
#include "params.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <queue>
#include <mutex>
#include <atomic>

namespace handy::camera {

struct StampedImagePtr {
    uint8_t* buffer = nullptr;
    tSdkFrameHead frame_info;
    uint64_t timestamp_nanosec;
};

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();

    static constexpr int NUM_OF_BUFFERS = 5;

  private:
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
        bool hardware_trigger = false;
        int max_buffer_size = 0;
    } param_{};

    struct State {
        std::vector<boost::lockfree::queue<StampedImagePtr>*> camera_images = {};
        std::vector<std::mutex> camera_bgr_buffer_mutexes = {};
    } state_{};

    std::vector<int> camera_handles_ = {};
    std::map<int, int> camera_idxs = {};
    std::unique_ptr<uint8_t[]> bgr_buffer_ = nullptr;
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
        rclcpp::CallbackGroup::SharedPtr handling_queue_timer = nullptr;
    } call_group_{};

    struct Timers {
        rclcpp::TimerBase::SharedPtr camera_soft_trigger = nullptr;
        rclcpp::TimerBase::SharedPtr camera_handle_queue_timer = nullptr;
    } timer_{};
};

}  // namespace handy::camera
