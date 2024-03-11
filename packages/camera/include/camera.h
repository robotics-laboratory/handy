#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/core/core.hpp>

#include "CameraApi.h"
#include "params.h"
#include "lock_free_queue.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace handy::camera {

struct StampedImageBufferId {
    size_t buffer_id = 0;
    tSdkFrameHead frame_info{};
    rclcpp::Time timestamp{};
};

struct CameraPool {
  public:
    CameraPool() = default;
    CameraPool(size_t height, size_t width, size_t frame_n)
        : frame_n_(frame_n), raw_frame_size_(height * width), bgr_frame_size_(height * width * 3) {
        raw_.resize(raw_frame_size_ * frame_n_);
        bgr_.resize(bgr_frame_size_ * frame_n_);
    }

    uint8_t* getRawFrame(size_t frame_idx) { return raw_.data() + frame_idx * raw_frame_size_; }
    uint8_t* getBgrFrame(size_t frame_idx) { return bgr_.data() + frame_idx * bgr_frame_size_; }

  private:
    size_t frame_n_ = 0;
    size_t raw_frame_size_ = 0;
    size_t bgr_frame_size_ = 0;
    std::vector<uint8_t> raw_ = {};
    std::vector<uint8_t> bgr_ = {};
};

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();

    constexpr static int MAX_CAMERA_NUM = 4;
    constexpr static int QUEUE_CAPACITY = 5;

  private:
    void applyParamsToCamera(int camera_idx);
    int getCameraId(int camera_handle);

    void triggerOnTimer();
    void handleFrame(CameraHandle idx, BYTE* raw_buffer, tSdkFrameHead* frame_info);
    void handleQueue(int camera_idx);

    void publishRawImage(uint8_t* buffer, const rclcpp::Time& timestamp, int camera_idx);
    void publishBGRImage(
        uint8_t* buffer, uint8_t* bgr_buffer, const rclcpp::Time& timestamp, int camera_idx,
        tSdkFrameHead& frame_inf);

    void abortIfNot(std::string_view msg, int status);
    void abortIfNot(std::string_view msg, int camera_idx, int status);

    struct Params {
        cv::Size preview_frame_size = cv::Size(640, 480);
        std::chrono::duration<double> latency{50.0};
        std::string calibration_file_path = "";
        int camera_num = MAX_CAMERA_NUM;
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
        std::array<std::unique_ptr<LockFreeQueue<StampedImageBufferId>>, MAX_CAMERA_NUM>
            camera_images;
        std::array<std::unique_ptr<LockFreeQueue<size_t>>, MAX_CAMERA_NUM> free_raw_buffer;
        std::vector<int> camera_handles = {};
        std::map<int, int> handle_to_camera_idx = {};
        std::vector<CameraIntrinsicParameters> cameras_intrinsics = {};
        std::vector<cv::Size> frame_sizes = {};
        CameraPool buffers;
    } state_{};

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
        std::vector<rclcpp::TimerBase::SharedPtr> camera_handle_queue_timer;
    } timer_{};
};

}  // namespace handy::camera
