#include <chrono>
#include <memory>
#include <vector>

#include <librealsense2/rs.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/boundingbox2d.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

using std::placeholders::_1;

class DetectorNode : public rclcpp::Node
{
    public:
        MinimalSubscriber() : Node("minimal_subscriber")
        {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "imgtopic", 10, std::bind(&DetectorNode::topic_callback, this, _1));
        publisher_ = this->create_publisher<vision_msgs::msg::BoundingBox2D>("topic", 10);
        }

    private:
        void imgtopic_callback(const sensor_msgs::msg::Image & img_msg) const
        {
            CvImageConstPtr cv_image = toCvShare(img_msg, "bgr8");
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
            auto bbox = vision_msgs::msg::BoundingBox2D();
            bbox.center.x = min_rect.x + min_rect.width/2.0;
            bbox.center.y = min_rect.y + min_rect.height/2.0;
            bbox.size_x = min_rect.width;
            bbox.size_y = min_rect.height;
            publisher_->publish(bbox);
        }
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
        rclcpp::Publisher<vision_msgs::msg::BoundingBox2D>::SharedPtr publisher_;

        int minHue = 22, maxHue = 30;
        int minSat = 120, maxSat = 255;
        int minVal = 140, maxVal = 255;
};

int main(int argc, char * argv[])
{
rclcpp::init(argc, argv);
rclcpp::spin(std::make_shared<MinimalSubscriber>());
rclcpp::shutdown();
return 0;
}