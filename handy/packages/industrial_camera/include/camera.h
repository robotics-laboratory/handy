#include "CameraApi.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <fstream>
#include <stdint.h>

using namespace std::chrono_literals;

constexpr unsigned int MAX_NUM_OF_CAMERAS = 10;

class CameraNode : public rclcpp::Node {
    public:
        CameraNode();
        ~CameraNode();

    
    private:
        int num_of_cameras_ = MAX_NUM_OF_CAMERAS;
        int camera_handles_[MAX_NUM_OF_CAMERAS];
        BYTE *raw_buffer_[MAX_NUM_OF_CAMERAS];
        BYTE *converted_buffer_[MAX_NUM_OF_CAMERAS];
        tSdkFrameHead frame_info_[MAX_NUM_OF_CAMERAS];
        rclcpp::Time last_frame_timestamps_[MAX_NUM_OF_CAMERAS];

        rclcpp::Clock::SharedPtr clock_;


        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr converted_img_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_img_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr small_preview_img_pub_;

        rclcpp::TimerBase::SharedPtr timer_;
        cv::Size frame_size_;
        cv::Size preview_frame_size_;

        void loadCalibrationParams(std::string &path);
        void applyDistortion();
        void applyCameraParameters(int camera_id);

        void allocateBuffersMemory();

        void publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);
        void publishConvertedImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id, bool publish_preview);
        void publishPreviewImage(cv::Mat &converted_image, rclcpp::Time timestamp, int camera_id);
        void publishPreviewImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id);

        std_msgs::msg::Header getHeader(rclcpp::Time timestamp, int camera_id);

        void handleCameraOnTimer();

        int getHandle(int i);

};
