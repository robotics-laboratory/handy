#include "camera.h"

CameraNode::CameraNode() : Node("camera_node") {
    int st = CameraSdkInit(0);
    RCLCPP_INFO_STREAM(this->get_logger(), "Init status: " << st);
    tSdkCameraDevInfo cameras_list;
    CameraEnumerateDevice(&cameras_list, &param_.num_of_cameras);
    assert(param_.num_of_cameras == this->declare_parameter<int>("num_of_cameras", 1));

    RCLCPP_INFO_STREAM(this->get_logger(), "Num of cameras attached: " << param_.num_of_cameras);

    camera_handles_ = std::vector<int>(param_.num_of_cameras);
    raw_buffer_ = std::vector<BYTE *>(param_.num_of_cameras);
    bgr_buffer_ = std::vector<BYTE *>(param_.num_of_cameras);
    frame_info_ = std::vector<tSdkFrameHead>(param_.num_of_cameras);
    state_.last_frame_timestamps = std::vector<rclcpp::Time>(param_.num_of_cameras);

    int status = CameraInit(&cameras_list, -1, -1, camera_handles_.data());
    if (status != CAMERA_STATUS_SUCCESS) {
        RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR occured during cameras initialisation, code: " << status);
        abort();
    }

    for (int i = 0; i < param_.num_of_cameras; ++i) {
        int status = CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8);
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Camera " << i << " couldn't set media format with code: " << status);
            abort();
        }

        status = CameraPlay(camera_handles_[i]);
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "Camera " << i << " failed to play with code: " << status);
            abort();
        }
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "Cameras started");

    param_.publish_bgr_topic = this->declare_parameter<bool>("publish_bgr_topic", false);
    param_.publish_bgr_preview = this->declare_parameter<bool>("publish_bgr_preview", false);
    param_.publish_raw_topic = this->declare_parameter<bool>("publish_raw_topic", false);
    param_.publish_raw_preview = this->declare_parameter<bool>("publish_raw_preview", false);

    for (int i = 0; i < param_.num_of_cameras; ++i) {
        if (param_.publish_raw_topic) {
            signals_.raw_img.push_back(this->create_publisher<sensor_msgs::msg::Image>("/camera_" + std::to_string(i + 1) + "/raw/image", 50));
        }
        if (param_.publish_raw_preview) {
            signals_.raw_preview_img.push_back(this->create_publisher<sensor_msgs::msg::CompressedImage>("/camera_" + std::to_string(i + 1) + "/raw/preview", 50));
        }
        if (param_.publish_bgr_topic) {
            signals_.bgr_img.push_back(this->create_publisher<sensor_msgs::msg::Image>("/camera_" + std::to_string(i + 1) + "/bgr/image", 50));
        }
        if (param_.publish_bgr_preview) {
            signals_.bgr_preview_img.push_back(this->create_publisher<sensor_msgs::msg::CompressedImage>("/camera_" + std::to_string(i + 1) + "/bgr/preview", 50));
        }
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "Publishers created");

    allocateBuffersMemory();
    RCLCPP_INFO_STREAM(this->get_logger(), "Buffer allocated");

    applyCameraParameters();

    int desired_fps = this->declare_parameter<int>("fps", 20);
    param_.latency = 1000 / desired_fps;
    warnLatency(param_.latency);
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(param_.latency), std::bind(&CameraNode::handleCameraOnTimer, this));
    RCLCPP_INFO_STREAM(this->get_logger(), "Cameras timer started with latency of " << param_.latency << "ms");

    RCLCPP_INFO_STREAM(this->get_logger(), "Initialisation finished");

}

void CameraNode::warnLatency(int latency) {
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        double d_num = latency;
        CameraGetExposureTime(camera_handles_[i], &d_num);
        if (d_num / 1000 > latency) {
            RCLCPP_WARN_STREAM(
                this->get_logger(),
                "Desired FPS is not possible due to exposure of " << d_num / 1000 << "ms exposure of camera " << i);
        }
    }
}

CameraNode::~CameraNode() {
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        delete[] bgr_buffer_[i];
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "Uninit done");
}

void CameraNode::allocateBuffersMemory() {
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        bgr_buffer_[i] = new uint8_t[param_.frame_size.height * param_.frame_size.width * 3];
    }
}

void CameraNode::publishBGRImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id) {
    int state =
        CameraImageProcess(camera_handles_[camera_id], buffer, bgr_buffer_[camera_id], &frame_info_[camera_id]);

    cv::Mat cv_image(std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth},
                     CV_8UC3,
                     bgr_buffer_[camera_id]);
    cv_bridge::CvImage cv_bridge_img(getHeader(timestamp, camera_id), "bgr8", cv_image);

    if (param_.publish_bgr_topic) {
        sensor_msgs::msg::Image img_msg;
        cv_bridge_img.toImageMsg(img_msg);
        signals_.bgr_img[camera_id]->publish(img_msg);
    }

    if (param_.publish_bgr_preview) {
        sensor_msgs::msg::CompressedImage comp_img_msg;
        //cv::resize(cv_img.image, cv_img.image, param_.preview_frame_size);
        cv_bridge_img.toCompressedImageMsg(comp_img_msg);
        signals_.bgr_preview_img[camera_id]->publish(comp_img_msg);
    }
}

void CameraNode::publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id) {
    cv::Mat cv_image(std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth}, CV_8UC1, buffer);
    cv_bridge::CvImage cv_img(getHeader(timestamp, camera_id), "mono8", cv_image);

    if (param_.publish_raw_topic) {
        sensor_msgs::msg::Image img_msg;
        cv_img.toImageMsg(img_msg);
        signals_.raw_img[camera_id]->publish(img_msg);
    }

    if (param_.publish_raw_preview) {
        sensor_msgs::msg::CompressedImage comp_img_msg;
        cv_img.toCompressedImageMsg(comp_img_msg);
        signals_.raw_preview_img[camera_id]->publish(comp_img_msg);
    }
}

void CameraNode::handleCameraOnTimer() {
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        state_.last_frame_timestamps[i] = this->get_clock()->now();
    }
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        int status = CameraGetImageBuffer(camera_handles_[i], &frame_info_[i], &raw_buffer_[i], 50);

        if (status == CAMERA_STATUS_TIME_OUT) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
            abort();
        } else if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR occured in handleCameraOnTimer, error code: " << status);
            abort();
        }
    }
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        if (param_.publish_raw_topic || param_.publish_raw_preview) {
            publishRawImage(raw_buffer_[i], state_.last_frame_timestamps[i], i);
        }
        if (param_.publish_bgr_topic || param_.publish_raw_preview) {
            publishBGRImage(raw_buffer_[i], state_.last_frame_timestamps[i], i);
        }
    }
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        CameraReleaseImageBuffer(camera_handles_[i], raw_buffer_[i]);
    }
}

std_msgs::msg::Header CameraNode::getHeader(rclcpp::Time timestamp, int camera_id) {
    std_msgs::msg::Header header;
    header.stamp = timestamp;
    header.frame_id = "camera_" + std::to_string(camera_id);

    return header;
}

void CameraNode::applyCameraParameters() {
    for (int i = 0; i < param_.num_of_cameras; ++i) {
        applyParamsToCamera(i);
    }
}

void CameraNode::applyParamsToCamera(int camera_idx) {
    int handle = camera_handles_[camera_idx];
    std::string prefix = "camera_" + std::to_string(camera_idx + 1) + ".";

    {
        const std::string param = "exposure_time";
        const std::string full_param = prefix + param;
        int status = CameraSetExposureTime(handle, this->declare_parameter<double>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        double value;
        CameraGetExposureTime(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Set \"" << param << "\" to " << value);
    }

    {
        const std::string param = "contrast";
        const std::string full_param = prefix + param;
        int status = CameraSetContrast(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        int value;
        CameraGetContrast(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Set \"" << param << "\" to " << value);
    }

    {
        const std::string param = "gain_rgb";
        const std::string full_param = prefix + param;
        const std::vector<int64_t> gain = this->declare_parameter<std::vector<int>>(full_param);
        int status = CameraSetGain(handle, gain[0], gain[1], gain[2]);
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        int value_r, value_g, value_b;
        CameraGetGain(handle, &value_r, &value_g, &value_b);
        RCLCPP_INFO_STREAM(this->get_logger(),
                           "Set \"" << param << "\" to (" << value_r << ", " << value_g << ", " << value_b << ')');
    }

    {
        const std::string param = "gamma";
        const std::string full_param = prefix + param;
        int status = CameraSetGamma(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        int value;
        CameraGetGamma(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Set \"" << param << "\" to " << value);
    }

    {
        const std::string param = "saturation";
        const std::string full_param = prefix + param;
        int status = CameraSetSaturation(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        int value;
        CameraGetSaturation(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Set \"" << param << "\" to " << value);
    }

    {
        const std::string param = "sharpness";
        const std::string full_param = prefix + param;
        int status = CameraSetSharpness(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        int value;
        CameraGetSharpness(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Set \"" << param << "\" to " << value);
    }

    {
        const std::string param = "auto_exposure";
        const std::string full_param = prefix + param;
        bool auto_exposure = this->declare_parameter<bool>(full_param);
        int status;
        if (auto_exposure) {
            status = CameraSetAeThreshold(handle, 10);
        } else {
            status = CameraSetAeThreshold(handle, 1000);
        }

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Set \"" << param << "\" for camera " << camera_idx << " failed: " << status);
            abort();
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "Set \"" << param << "\" to " << auto_exposure);
    }
}