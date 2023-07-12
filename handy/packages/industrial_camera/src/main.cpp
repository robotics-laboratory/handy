#include "camera.h"
#include "CameraApi.h"


CameraNode::CameraNode() : Node("main_camera_node") {

    int st = CameraSdkInit(0);
    RCLCPP_INFO_STREAM(this->get_logger(), "init status: " << st);
    tSdkCameraDevInfo cameras_list;
    CameraEnumerateDevice(&cameras_list, &num_of_cameras_);

    RCLCPP_INFO_STREAM(this->get_logger(), "Num of cameras attached: " << num_of_cameras_);

    int status = CameraInit(&cameras_list, -1, -1, camera_handles_);
    if (status != CAMERA_STATUS_SUCCESS) {
        RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR occured during cameras initialisation, code: " << status);
    }

    for (size_t i = 0; i < num_of_cameras_; ++i) {
        CameraPlay(camera_handles_[i]);
        CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "cameras started");


    raw_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/raw_image", 10);
    converted_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/converted_image", 10);
    small_preview_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/preview_image", 10);


    frame_size_ = cv::Size(1280, 1024);
    preview_frame_size_ = cv::Size(640, 480);

    RCLCPP_INFO_STREAM(this->get_logger(), "publishers created");
    allocateBuffersMemory();
    RCLCPP_INFO_STREAM(this->get_logger(), "buffer allocated");

    timer_ = this->create_wall_timer(1000ms, std::bind(&CameraNode::handleCameraOnTimer, this));
    clock_ = this->get_clock();

    RCLCPP_INFO_STREAM(this->get_logger(), "init done");

}

CameraNode::~CameraNode() {
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        free(converted_buffer_[i]);
        //free(raw_buffer_[i]);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "uninit done");
}

void CameraNode::allocateBuffersMemory() {
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        raw_buffer_[i] = (unsigned char *)malloc(frame_size_.height * frame_size_.width);
        converted_buffer_[i] = (unsigned char *)malloc(frame_size_.height * frame_size_.width * 3);
    }
}

int CameraNode::getHandle(int i) {
    return camera_handles_[i];
}

void CameraNode::publishConvertedImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id, bool publish_preview=false) {
    int state = CameraImageProcess(camera_handles_[camera_id], buffer, converted_buffer_[camera_id], &frame_info_[camera_id]);

    cv::Mat cv_image(std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth}, CV_8UC3, converted_buffer_[camera_id]);
    sensor_msgs::msg::Image img_msg;
    
    cv_bridge::CvImage cv_img(getHeader(timestamp, camera_id), "bgr8", cv_image);
    cv_img.toImageMsg(img_msg);
    converted_img_pub_->publish(img_msg);
    if (publish_preview) {
        cv::resize(cv_img.image, cv_img.image, preview_frame_size_);
        cv_img.toImageMsg(img_msg);
        small_preview_img_pub_->publish(img_msg);
    }

}

void CameraNode::publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id) {
    cv::Mat cv_image(std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth}, CV_8UC1, buffer);
    sensor_msgs::msg::Image img_msg;
    
    cv_bridge::CvImage cv_img(getHeader(timestamp, camera_id), "8uc1", cv_image);
    cv_img.toImageMsg(img_msg);
    raw_img_pub_->publish(img_msg);

}

void CameraNode::handleCameraOnTimer() {
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        int status = CameraGetImageBuffer(camera_handles_[i], &frame_info_[i], &raw_buffer_[i], 100);
        if (status == CAMERA_STATUS_TIME_OUT){
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
            return;
        }
        else if (status != CAMERA_STATUS_SUCCESS){
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR occured in handleCameraOnTimer, error code: " << status);
            return;
        }
        last_frame_timestamps_[i] = clock_->now();

    }
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        publishRawImage(raw_buffer_[i], last_frame_timestamps_[i], i);
        publishConvertedImage(raw_buffer_[i], last_frame_timestamps_[i], i, true);
    }
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        CameraReleaseImageBuffer(camera_handles_[i], raw_buffer_[i]);
    }



}

std_msgs::msg::Header CameraNode::getHeader(rclcpp::Time timestamp, int camera_id) {
    std_msgs::msg::Header header;
    header.stamp = timestamp;
    header.frame_id = camera_id;

    return header;
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();

    return 0;

}