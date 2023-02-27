#include <chrono>
#include <memory>
#include <vector>
#include <string>

#include <librealsense2/rs.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/image_marker.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

using std::placeholders::_1;



class DetectorNode : public rclcpp::Node
{

    public:
        DetectorNode() : Node("detector")
        {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/device_0/sensor_1/Color_0/image/data", 10, std::bind(&DetectorNode::imgtopic_callback, this, _1));
        publisher_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("detection", 10);
        }

    private:
        void imgtopic_callback(const sensor_msgs::msg::Image & img_msg) const
        {
            const std::string type = "bgr8";
            cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(img_msg, type);
            cv::Mat imageHSV;
            cv::cvtColor(cv_image->image, imageHSV, cv::COLOR_BGR2HSV);
            cv::Mat mask;
            cv::inRange(
                imageHSV, 
                cv::Scalar(minHue, minSat, minVal), 
                cv::Scalar(maxHue, maxSat, maxVal), 
                mask
            );
            cv::Mat nonZeroCoordinates;
            cv::findNonZero(mask, nonZeroCoordinates);
            cv::Rect min_rect = cv::boundingRect(nonZeroCoordinates);
            auto bbox = visualization_msgs::msg::ImageMarker();
            bbox.id = 0;
            bbox.type = 3;
            bbox.action = 3;

            geometry_msgs::msg::Point a, b, c, d;
            a.x = min_rect.x;
            a.y = min_rect.y;
            b.x = min_rect.x + min_rect.width;
            b.y = min_rect.y;
            c.x = min_rect.x + min_rect.width;
            c.y = min_rect.y + min_rect.height;
            d.x = min_rect.x;
            d.y =  min_rect.y + min_rect.height;
            bbox.points.push_back(a);
            bbox.points.push_back(b);
            bbox.points.push_back(c);
            bbox.points.push_back(d);
            publisher_->publish(bbox);
        }
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
        rclcpp::Publisher<visualization_msgs::msg::ImageMarker>::SharedPtr publisher_;

        int minHue = 22, maxHue = 30;
        int minSat = 120, maxSat = 255;
        int minVal = 140, maxVal = 255;
};

int main(int argc, char * argv[])
{
rclcpp::init(argc, argv);
rclcpp::spin(std::make_shared<DetectorNode>());
rclcpp::shutdown();
return 0;
}