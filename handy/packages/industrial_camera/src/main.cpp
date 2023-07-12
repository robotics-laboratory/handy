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
        RCLCPP_INFO_STREAM(this->get_logger(), "ERROR occured during cameras initialisation, code: " << status);
    }

    for (size_t i = 0; i < num_of_cameras_; ++i) {
        CameraPlay(camera_handles_[i]);
        CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "cameras started");


    raw_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/raw_image", 10);
    converted_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/converted_image", 10);
    small_preview_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/preview_image", 10);

    RCLCPP_INFO_STREAM(this->get_logger(), "publishers created");
    allocateBuffersMemory();
    RCLCPP_INFO_STREAM(this->get_logger(), "buffer allocated");

    timer_ = this->create_wall_timer(1000ms, std::bind(&CameraNode::handleCameraOnTimer, this));
    clock_ = this->get_clock();

    frame_size_ = cv::Size(1280, 1024);
    RCLCPP_INFO_STREAM(this->get_logger(), "init done");

}

CameraNode::~CameraNode() {
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        free(converted_buffer_[i]);
    }

}

void CameraNode::allocateBuffersMemory() {
    for (size_t i = 0; i < 10; ++i) {
        raw_buffer_[i] = (BYTE *)malloc(frame_size_.height * frame_size_.width + 1000);
        converted_buffer_[i] = (BYTE *)malloc(frame_size_.height * frame_size_.width * 3 + 3000);
        //frame_info[i] = (tSdkFrameHead *)malloc(200);
    }
}

int CameraNode::getHandle(int i) {
    return camera_handles_[i];
}

void CameraNode::publishConvertedImage(BYTE **buffer, rclcpp::Time timestamp, int camera_id, bool publish_preview=false) {
    RCLCPP_INFO_STREAM(this->get_logger(), "converting buffer... camera_id:  " << camera_id);

    RCLCPP_INFO_STREAM(this->get_logger(), "" << frame_info_[camera_id].iHeight << ' ' << frame_info_[camera_id].iWidth);
    int state;
    try {
        state = CameraImageProcess(camera_handles_[camera_id], *buffer, converted_buffer_[camera_id], &frame_info_[camera_id]);
    }
    catch (const std::exception &e) {
        RCLCPP_INFO_STREAM(this->get_logger(), e.what());
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "converted buffer with code: " << state);
    RCLCPP_INFO_STREAM(this->get_logger(), "converted buffer  " << frame_info_[camera_id].iHeight << ' ' << frame_info_[camera_id].iWidth);
    cv::Mat cv_image(std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth}, CV_8UC3, converted_buffer_[camera_id]);
    cv::imwrite("test.png", cv_image);

    sensor_msgs::msg::Image img_msg;
    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg(img_msg);
    converted_img_pub_->publish(img_msg);
    RCLCPP_INFO(this->get_logger(), "image sent");

}

void CameraNode::publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id) {
    
}

void CameraNode::handleCameraOnTimer() {
    RCLCPP_INFO_STREAM(this->get_logger(), "handling camera on timer");
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        int status = CameraGetImageBuffer(camera_handles_[i], &frame_info_[i], &raw_buffer_[i], 100);
        RCLCPP_INFO_STREAM(this->get_logger(), "raw buffer got with code: " << status);
        if (status == CAMERA_STATUS_TIME_OUT){
            RCLCPP_INFO_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
            return;
        }
        else if (status != CAMERA_STATUS_SUCCESS){
            RCLCPP_INFO_STREAM(this->get_logger(), "ERROR occured in handleCameraOnTimer, error code: " << status);
            return;
        }
        last_frame_timestamps_[i] = clock_->now();
        RCLCPP_INFO_STREAM(this->get_logger(), "" << frame_info_[i].iHeight << ' ' << frame_info_[i].iWidth);
        RCLCPP_INFO_STREAM(this->get_logger(), "writing...");
        std::ofstream output("img.raw", std::ios::binary);

        for (size_t j = 0; j < 1280 * 1024; ++j) {
            output << *(raw_buffer_[i] + j);
            if (j == 1280 * 1024 - 1) {
                RCLCPP_INFO_STREAM(this->get_logger(), "last iter...");
            }
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "finished...");
        CameraImageProcess(camera_handles_[i], raw_buffer_[i], converted_buffer_[i], &frame_info_[i]);
    }
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        RCLCPP_INFO_STREAM(this->get_logger(), "publishing...");
        publishConvertedImage(&raw_buffer_[i], last_frame_timestamps_[i], i);
        RCLCPP_INFO_STREAM(this->get_logger(), "published");
    }
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        CameraReleaseImageBuffer(camera_handles_[i], raw_buffer_[i]);
    }



}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();

    return 0;

}