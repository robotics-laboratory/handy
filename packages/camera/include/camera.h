#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/core/core.hpp>
#include <boost/lockfree/queue.hpp>

#include "CameraApi.h"
#include "params.h"
#include "lock_free_queue.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace handy::camera {

struct StampedImagePtr {
    std::shared_ptr<uint8_t[]> buffer = nullptr;
    tSdkFrameHead frame_info{};
    rclcpp::Time timestamp{};
};

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();

    constexpr static int QUEUE_CAPACITY = 8;
    constexpr static int MAX_CAMERA_NUM = 10;

  private:
    void applyParamsToCamera(int camera_idx);
    int getCameraId(int camera_handle);

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
        int camera_num = MAX_CAMERA_NUM;  // 10
        bool publish_bgr = false;
        bool publish_bgr_preview = false;
        bool publish_raw = false;
        bool publish_raw_preview = false;
        bool publish_rectified_preview = false;
        bool hardware_trigger = false;
        int max_buffer_size = 0;
        int master_camera_idx = -1;
    } param_{};

    struct State {
        std::array<std::unique_ptr<LockFreeQueue<StampedImagePtr>>, MAX_CAMERA_NUM> camera_images;
    } state_{};

    std::vector<int> camera_handles_ = {};
    std::map<int, int> handle_to_camera_idx = {};
    LockFreeQueue<std::shared_ptr<uint8_t[]>> free_bgr_buffer_;
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
