#pragma once

#include "CameraApi.h"

#include "opencv2/highgui/highgui.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdint.h>
#include <string>

using namespace std::chrono_literals;

constexpr unsigned int MAX_NUM_OF_CAMERAS = 10;

class CameraNode : public rclcpp::Node {
  public:
    CameraNode();
    ~CameraNode();

  private:
    void loadCalibrationParams(std::string &path);
    void applyDistortion();
    void applyCameraParameters();
    void applyParamsToCamera(
        int camera_handle, const std::string &param_type, std::string &param_name);
    void handleParamStatus(
        int camera_handle, const std::string &param_type, const std::string &param_name,
        int status);

    void allocateBuffersMemory();

    void publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);
    void publishConvertedImage(
        BYTE *buffer, rclcpp::Time timestamp, int camera_id, bool publish_preview);
    void publishPreviewImage(cv::Mat &converted_image, rclcpp::Time timestamp, int camera_id);
    void publishPreviewImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);

    std_msgs::msg::Header getHeader(rclcpp::Time timestamp, int camera_id);

    void handleCameraOnTimer();

    int getHandle(int i);

    int num_of_cameras_ = MAX_NUM_OF_CAMERAS;
    int camera_handles_[MAX_NUM_OF_CAMERAS];
    BYTE *raw_buffer_[MAX_NUM_OF_CAMERAS];
    BYTE *converted_buffer_[MAX_NUM_OF_CAMERAS];
    tSdkFrameHead frame_info_[MAX_NUM_OF_CAMERAS];
    rclcpp::Time last_frame_timestamps_[MAX_NUM_OF_CAMERAS];

    struct Signals {
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr converted_img_pub = nullptr;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_img_pub = nullptr;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr small_preview_img_pub = nullptr;
    } signals_;

    rclcpp::TimerBase::SharedPtr timer_ = nullptr;
    cv::Size frame_size_;
    cv::Size preview_frame_size_;
};
