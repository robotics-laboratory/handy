#include <chrono>
#include <memory>
#include <vector>
#include <string>

#include <librealsense2/rs.hpp>
#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include "visualization_msgs/msg/image_marker.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using std::placeholders::_1;

const static int kCV_64FSize = 8;



class DetectorNode : public rclcpp::Node
{

    public:
        DetectorNode() : Node("detector"), P_(3, 4, CV_64F)
        {
        subscription_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "imgtopic", 10, std::bind(&DetectorNode::imgtopic_callback, this, _1));
        subscription_params_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/device_0/sensor_1/Color_0/info/camera_info", 10, std::bind(&DetectorNode::info_callback, this, _1));
        publisher_bbox_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("detection", 10);
        publisher_pose_ = this->create_publisher<geometry_msgs::msg::Pose>("/ball/pose", 10);
        publisher_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("/ball/visualization", 10);
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
            bbox.action = 0;
            bbox.filled = 1;
            bbox.outline_color.r = 1;
            bbox.outline_color.a = 1;
            bbox.fill_color.r = 1;
            bbox.fill_color.a = 0.5;
            bbox.lifetime.nanosec = 100;

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
            publisher_bbox_->publish(bbox);

            if (P_.empty()) {
                cv::Mat inversed_P(4, 3, CV_32FC1);
                cv::invert(P_, inversed_P);
                cv::Vec3f bottom_point = {min_rect.x + min_rect.width/2, min_rect.y, 1}, 
                            top_point = {min_rect.x + min_rect.width/2, min_rect.y + min_rect.height, 1};
                cv::Vec4f bottom_beam = inversed_P.dot(bottom_point), top_beam = inversed_P.dot(top_point);
                float k = (bottom_beam[1] - top_beam[1]) / BALL_WIDTH;
                cv::Vec4f ball_point = (bottom_beam + top_beam) * (k/2);

                geometry_msgs::msg::Pose pose;
                pose.position.x = ball_point[0];
                pose.position.y = ball_point[1];
                pose.position.z = ball_point[2];
                pose.orientation.w = 1;

                visualization_msgs::msg::Marker marker;
                marker.id = 1;
                marker.type = 2;
                marker.action = 0;
                marker.pose = pose;
                marker.color = bbox.fill_color;
                marker.lifetime = bbox.lifetime;
                publisher_pose_->publish(pose);
                publisher_marker_->publish(marker);
            }
        }

        void info_callback(const sensor_msgs::msg::CameraInfo& info_msg)
        {
            std::memcpy(P_.data, info_msg.p.data(), 3 * 4 * kCV_64FSize);
        }

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_image_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription_params_;
        rclcpp::Publisher<visualization_msgs::msg::ImageMarker>::SharedPtr publisher_bbox_;
        rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_pose_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_marker_;
        cv::Mat P_;

        int minHue = 22, maxHue = 30;
        int minSat = 120, maxSat = 255;
        int minVal = 140, maxVal = 255;
        float BALL_WIDTH = 4;
};

int main(int argc, char * argv[])
{
rclcpp::init(argc, argv);
rclcpp::spin(std::make_shared<DetectorNode>());
rclcpp::shutdown();
return 0;
}