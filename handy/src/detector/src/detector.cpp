

#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <deque>

#include <librealsense2/rs.hpp>
#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include "visualization_msgs/msg/image_marker.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using std::placeholders::_1;


void cutRect(cv::Rect& rect, int cols, int rows) {
    if (rect.x < 0) {
        rect.width += rect.x;
        rect.x = 0;
    }

    if (rect.y < 0) {
        rect.width += rect.y;
        rect.y = 0;
    }

    if (rect.x + rect.width > cols) {
        rect.width = cols - rect.x;
    }

    if (rect.y + rect.height > rows) {
        rect.height = rows - rect.y;
    }

    if (rect.width < 0) {
        rect.width = 0;
    }

    if (rect.height < 0) {
        rect.height = 0;
    }
}

cv::Rect resize(cv::Rect& rect, float coef, int cols, int rows) {
    cv::Rect result;
    if (rect.width * (coef - 1) <= 20) {
        result = cv::Rect(rect.x - 10, rect.y - 10, rect.width + 20, rect.height + 20);
    } else {
        result = cv::Rect(cvRound(rect.x - rect.width * (coef - 1) * 0.5), cvRound(rect.y - rect.height * (coef - 1) * 0.5), 
                                        cvRound(rect.width * coef), cvRound(rect.height * coef));
    }

    cutRect(result, cols, rows);
    return result;
}

void publish_bbox(rclcpp::Publisher<visualization_msgs::msg::ImageMarker>::SharedPtr publisher, cv::Rect rect, float r_c, float b_c, float g_c, int id, const sensor_msgs::msg::Image& img_msg) {
    auto bbox = visualization_msgs::msg::ImageMarker();
    bbox.header.frame_id = "camera_color_optical_frame";
    bbox.header.stamp = img_msg.header.stamp;
    bbox.id = id;
    bbox.type = visualization_msgs::msg::ImageMarker::POLYGON;
    bbox.action = visualization_msgs::msg::ImageMarker::ADD;
    bbox.filled = 1;
    bbox.outline_color.r = r_c;
    bbox.outline_color.g = g_c;
    bbox.outline_color.b = b_c;
    bbox.outline_color.a = 1;
    bbox.fill_color.r = r_c;
    bbox.fill_color.g = g_c;
    bbox.fill_color.b = b_c;
    bbox.fill_color.a = 0.5;
    bbox.lifetime.nanosec = 100;

    geometry_msgs::msg::Point a, b, c, d;
    a.x = rect.x;
    a.y = rect.y;
    b.x = rect.x + rect.width;
    b.y = rect.y;
    c.x = rect.x + rect.width;
    c.y = rect.y + rect.height;
    d.x = rect.x;
    d.y =  rect.y + rect.height;
    bbox.points.push_back(a);
    bbox.points.push_back(b);
    bbox.points.push_back(c);
    bbox.points.push_back(d);
    publisher->publish(bbox);
}



class DetectorNode : public rclcpp::Node
{

    public:
        DetectorNode() : Node("detector"), P_(3, 4, CV_32FC1), inversed_P(4, 3, CV_32FC1)
        {
        subscription_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_raw", 10, std::bind(&DetectorNode::imgtopic_callback, this, _1));
        subscription_params_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/color/camera_info", 10, std::bind(&DetectorNode::info_callback, this, _1));
        publisher_bbox_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("/detection", 10);
        publisher_bbox_center_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("/detection_center", 10);
        publisher_bbox_hist_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("/history_roi", 10);
        publisher_bbox_color_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("/color_detection", 10);
        publisher_bbox_poss_ = this->create_publisher<visualization_msgs::msg::ImageMarker>("/possible_roi", 10);
        publisher_pose_ = this->create_publisher<geometry_msgs::msg::Pose>("/ball/pose", 10);
        publisher_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("/ball/visualization", 10);
        publisher_tf_ = this->create_publisher<tf2_msgs::msg::TFMessage>("/tf", 10);
        }

    private:


        void imgtopic_callback(const sensor_msgs::msg::Image & img_msg) 
        {
            const std::string type = "bgr8";
            cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(img_msg, type);
            cv::Mat imageHSV;
            cv::cvtColor(cv_image->image, imageHSV, cv::COLOR_BGR2HSV);
            cv::Mat gray;
            cv::cvtColor(cv_image->image, gray, cv::COLOR_BGR2GRAY);


            cv::Rect history_based_roi(0, 0, gray.cols, gray.rows);
            if (prev_detection.size() == HISTORY_SIZE) {
                cv::Rect prev_frame = prev_detection.back();
                cv::Rect last_frame = prev_detection.front();
                cv::Point prev_center(cvRound(prev_frame.x + prev_frame.width * 0.5), cvRound(prev_frame.y + prev_frame.height * 0.5));

                std::vector<cv::Point> corners = {{last_frame.x, last_frame.y}, {last_frame.x + last_frame.width, last_frame.y}, 
                                    {last_frame.x + last_frame.width, last_frame.y + last_frame.height}, {last_frame.x, last_frame.y + last_frame.height}};
                cv::Point shift;
                float dist = 0;
                for (const auto &c : corners) {
                    if (cv::norm(c - prev_center) > dist) {
                        dist = cv::norm(c - prev_center);
                        shift = c - prev_center;
                    }
                }

                cv::Point corner1 = prev_center + shift, corner2 = prev_center - shift;
                cv::Rect borders(std::min(corner1.x, corner2.x), std::min(corner1.y, corner2.y), std::abs(corner1.x - corner2.x), std::abs(corner1.y - corner2.y));
                borders = resize(borders, 1.5, gray.cols, gray.rows);
                if (borders.width > 0 && borders.height > 0) {
                    history_based_roi = borders;
                }
                publish_bbox(publisher_bbox_hist_, history_based_roi, 0, 1, 0, 1, img_msg);
            }

            cv::Mat mask;
            cv::inRange(imageHSV(history_based_roi), cv::Scalar(minHue, minSat, minVal), cv::Scalar(maxHue, maxSat, maxVal), mask);
            cv::Mat nonZeroCoordinates;
            cv::findNonZero(mask, nonZeroCoordinates);
            cv::Rect min_rect = cv::boundingRect(nonZeroCoordinates);
            min_rect.x += history_based_roi.x;
            min_rect.y += history_based_roi.y;
            publish_bbox(publisher_bbox_color_, min_rect, 0, 0, 1, 2, img_msg);


            cv::Rect possible_roi = resize(min_rect, 2.5, gray.cols, gray.rows);
            publish_bbox(publisher_bbox_poss_, possible_roi, 0, 1, 1, 3, img_msg);
            cv::Point max_center(0, 0);
            int max_rad = 0;
            std::vector<cv::Vec3f> circles;
            if (!gray(possible_roi).empty()) {
                cv::HoughCircles(gray(possible_roi), circles, cv::HOUGH_GRADIENT_ALT, 1, std::max(1, std::min(min_rect.width, min_rect.height) / 10), 600, 0.85, std::max(1, std::min(min_rect.width, min_rect.height) / 3), std::max(1, cvRound(std::max(min_rect.width, min_rect.height) / 1.5)));
            }
            for (size_t i = 0; i < circles.size(); i++)
            {
                cv::Point center(cvRound(circles[i][0]) + possible_roi.x, cvRound(circles[i][1]) + possible_roi.y);
                int radius = cvRound(circles[i][2]);
                if (min_rect.x <= center.x && center.x <= min_rect.x + min_rect.width && 
                        min_rect.y <= center.y && center.y <= min_rect.y + min_rect.height) {
                    if (radius > max_rad) {
                        max_rad = radius;
                        max_center = center;
                    }
                }
            }

            if (max_rad > 0) {
                cv::Rect roi(max_center.x - max_rad, max_center.y - max_rad, 2 * max_rad, 2 * max_rad);
                min_rect = roi;
            }

            publish_bbox(publisher_bbox_, min_rect, 1, 0, 0, 0, img_msg);

            if (min_rect.width != 0 && min_rect.height != 0) {
                prev_detection.push_back(min_rect);
            } else {
                prev_detection.clear();
            }
            if (prev_detection.size() > HISTORY_SIZE) {
                prev_detection.pop_front();
            }

            if (!P_.empty()) {
                cv::Vec3f bottom_point = {min_rect.x + min_rect.width/2, min_rect.y, 1}, 
                            top_point = {min_rect.x + min_rect.width/2, min_rect.y + min_rect.height, 1};
                cv::Mat bottom_beam_mat = inversed_P * cv::Mat(bottom_point, CV_32FC1), top_beam_mat = inversed_P * cv::Mat(top_point, CV_32FC1);
                cv::Vec4f bottom_beam(bottom_beam_mat.reshape(4).at<cv::Vec4f>()), top_beam(top_beam_mat.reshape(4).at<cv::Vec4f>());
                bottom_beam /= bottom_beam[2];
                top_beam /= top_beam[2];
                float k = BALL_WIDTH / (top_beam[1] - bottom_beam[1]);
                cv::Vec4f ball_point = (bottom_beam + top_beam) * (k/2);

                auto bbox = visualization_msgs::msg::ImageMarker();
                bbox.header.frame_id = "camera_color_optical_frame";
                bbox.header.stamp = img_msg.header.stamp;
                bbox.id = 5;
                bbox.type = visualization_msgs::msg::ImageMarker::CIRCLE;
                bbox.action = visualization_msgs::msg::ImageMarker::ADD;
                bbox.scale = 1;
                bbox.filled = 1;
                bbox.outline_color.g = 1;
                bbox.outline_color.a = 1;
                bbox.fill_color.g = 1;
                bbox.fill_color.a = 1;
                bbox.lifetime.nanosec = 100;

                geometry_msgs::msg::Point center;
                center.x = min_rect.x + min_rect.width/2;
                center.y = min_rect.y + min_rect.height/2;
                bbox.position = center;
                publisher_bbox_center_->publish(bbox);

                geometry_msgs::msg::Pose pose;
                pose.position.x = ball_point[0];
                pose.position.y = ball_point[1];
                pose.position.z = ball_point[2];
                pose.orientation.w = 1;
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "camera_color_optical_frame";
                marker.id = 1;
                marker.type = 2;
                marker.action = 0;
                marker.pose = pose;
                marker.color.r = 1;
                marker.color.a = 1;
                marker.scale.x = 0.02;
                marker.scale.y = 0.02;
                marker.scale.z = 0.02;
                marker.lifetime.nanosec = 100;
                publisher_pose_->publish(pose);
                publisher_marker_->publish(marker);

                tf2_msgs::msg::TFMessage result;
                geometry_msgs::msg::TransformStamped msg;
                msg.header.frame_id = "camera_color_optical_frame";
                msg.child_frame_id = "ball";

                msg.transform.translation.x = ball_point[0];
                msg.transform.translation.y = ball_point[1];
                msg.transform.translation.z = ball_point[2];
                msg.transform.rotation = pose.orientation;
                result.transforms.push_back(msg);
                publisher_tf_->publish(result);
            }
        }

        void info_callback(const sensor_msgs::msg::CameraInfo& info_msg)
        {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    P_.at<float>(i,j) = info_msg.p[i * 4 + j];
                }
            }
            cv::invert(P_, inversed_P, cv::DECOMP_SVD);
        }

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_image_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription_params_;
        rclcpp::Publisher<visualization_msgs::msg::ImageMarker>::SharedPtr publisher_bbox_, publisher_bbox_hist_, publisher_bbox_color_, publisher_bbox_poss_, publisher_bbox_center_;
        rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_pose_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_marker_;
        rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr publisher_tf_;
        cv::Mat P_;
        cv::Mat inversed_P;

        std::deque<cv::Rect> prev_detection;


        int minHue = 50, maxHue = 80;
        int minSat = 90, maxSat = 255;
        int minVal = 105, maxVal = 255;
        float BALL_WIDTH = 0.04;
        size_t HISTORY_SIZE = 4;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectorNode>());
    rclcpp::shutdown();
    return 0;
}