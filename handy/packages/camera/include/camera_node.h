#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"

#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <sensor_msgs/msg/camera_info.hpp>

using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node {
    public:
        CameraNode();
    
    protected:
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_image_pub_;
        rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr test_pub;


        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::TimerBase::SharedPtr test_timer_;

        cv::VideoCapture camera_;
        int cnt = 0;


        void cameraHandler();

        void pub_next_value();


        bool publishImage(const cv::Mat &image);

        std_msgs::msg::Header constructHeader();



};