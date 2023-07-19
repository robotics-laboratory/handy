#pragma once

#include "CameraApi.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <fstream>
#include <vector>
#include <stdint.h>
#include <string>

using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();

    static const unsigned int MAX_NUM_OF_CAMERAS = 1;

  private:
    void applyCameraParameters();
    void applyParamsToCamera(int camera_idx);

    void warnLatency(int latency);
    void allocateBuffersMemory();
    void measureFPS();

    void publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);
    void publishConvertedImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);
    void publishPreviewImage(cv::Mat &converted_image, rclcpp::Time timestamp, int camera_id);
    void publishPreviewImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);

    std_msgs::msg::Header getHeader(rclcpp::Time timestamp, int camera_id);

    void handleCameraOnTimer();

    int getHandle(int i);

    int camera_handles_[MAX_NUM_OF_CAMERAS];
    BYTE *raw_buffer_[MAX_NUM_OF_CAMERAS];
    BYTE *converted_buffer_[MAX_NUM_OF_CAMERAS];
    tSdkFrameHead frame_info_[MAX_NUM_OF_CAMERAS];

    struct Signals {
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr converted_img_pub[MAX_NUM_OF_CAMERAS];
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_img_pub[MAX_NUM_OF_CAMERAS];
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
            small_preview_img_pub[MAX_NUM_OF_CAMERAS];

    } signals_;

    struct Params {
        cv::Size frame_size_ = cv::Size(1280, 1024);
        cv::Size preview_frame_size_ = cv::Size(640, 480);
        int num_of_cameras_ = MAX_NUM_OF_CAMERAS;
        bool publish_preview = false;

    } param_;

    struct State {
        int save_image_id_ = 0;
        int frame_id_on_save = 0;
        rclcpp::Time last_frame_timestamps_[MAX_NUM_OF_CAMERAS];

    } state_;

    rclcpp::TimerBase::SharedPtr timer_ = nullptr;
};
