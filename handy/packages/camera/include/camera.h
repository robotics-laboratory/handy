#pragma once

#include "CameraApi.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <stdint.h>
#include <string>

using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();

  private:
    void applyCameraParameters();
    void applyParamsToCamera(int camera_idx);

    void warnLatency(int latency);
    void allocateBuffersMemory();

    void publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);
    void publishBGRImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);
    void publishPreviewImage(cv::Mat &bgr_image, rclcpp::Time timestamp, int camera_id);
    void publishPreviewImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);

    std_msgs::msg::Header getHeader(rclcpp::Time timestamp, int camera_id);

    void handleCameraOnTimer();

    std::vector<int> camera_handles_;
    std::vector<BYTE *> raw_buffer_;
    std::vector<BYTE *> bgr_buffer_;
    std::vector<tSdkFrameHead> frame_info_;

    struct Signals {
        std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> raw_img;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> raw_preview_img;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> bgr_img;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> bgr_preview_img;
    } signals_;

    struct Params {
        cv::Size frame_size = cv::Size(1280, 1024);
        cv::Size preview_frame_size_ = cv::Size(640, 480);
        int num_of_cameras = 10;
        bool publish_bgr_topic = true;
        bool publish_bgr_preview = false;
        bool publish_raw_topic = true;
        bool publish_raw_preview = true;
        int latency = 33;
    } param_;

    struct State {
        std::vector<rclcpp::Time> last_frame_timestamps;
    } state_;

    rclcpp::TimerBase::SharedPtr timer_ = nullptr;
};
