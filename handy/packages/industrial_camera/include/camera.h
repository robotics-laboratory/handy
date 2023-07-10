#include "CameraApi.h"

#include <rclcpp/rclcpp.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <stdint.h>


constexpr unsigned int MAX_NUM_OF_CAMERAS = 10;

class CameraNode : public rclcpp::Node {
    public:
        CameraNode();
    
    private:
        int num_of_cameras;
        int camera_handles;
        BYTE *raw_buffer[MAX_NUM_OF_CAMERAS];
        //BYTE *converted_buffer[MAX_NUM_OF_CAMERAS];
        rclcpp::Logger logger_;


        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr converted_img_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_img_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr small_preview_img_pub_;

        rclcpp::TimerBase::SharedPtr timer_;
        cv::Size frame_size_;

        void loadCalibrationParams(std::string &path);
        void applyDistortion();
        void applyCameraParameters(int camera_handle);

        void allocateBuffersMemory();

        void publishRawImage(BYTE *buffer, uint32 timestamp, int camera_handle);
        void publishConvertedImage(BYTE *buffer, uint32 timestamp, int camera_handle, bool publish_preview=false);
        void publishPreviewImage(cv::Mat &converted_image, uint32 timestamp, int camera_handle);
        void publishPreviewImage(BYTE *buffer, uint32 timestamp, int camera_handle);

        void handleCameraOnTimer();

        int getHandle(int i);

};
