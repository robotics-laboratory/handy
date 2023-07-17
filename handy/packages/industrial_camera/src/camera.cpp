#include "camera.h"

CameraNode::CameraNode() : Node("camera_node") {
    int st = CameraSdkInit(0);
    RCLCPP_INFO_STREAM(this->get_logger(), "init status: " << st);
    tSdkCameraDevInfo cameras_list;
    CameraEnumerateDevice(&cameras_list, &param_.num_of_cameras_);

    RCLCPP_INFO_STREAM(this->get_logger(), "Num of cameras attached: " << param_.num_of_cameras_);

    int status = CameraInit(&cameras_list, -1, -1, camera_handles_);
    if (status != CAMERA_STATUS_SUCCESS) {
        RCLCPP_ERROR_STREAM(this->get_logger(),
                            "ERROR occured during cameras initialisation, code: " << status);
        abort();
    }

    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        int status = CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8);
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Camera " << i << " couldn't set media format with code: " << status);
            abort();
        }
        status = CameraPlay(camera_handles_[i]);
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Camera " << i << " failed to play with code: " << status);
            abort();
        }
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "cameras started");

    signals_.raw_img_pub = this->create_publisher<sensor_msgs::msg::Image>("/camera/raw_image", 10);
    signals_.converted_img_pub =
        this->create_publisher<sensor_msgs::msg::Image>("/camera/converted_image", 10);
    signals_.small_preview_img_pub =
        this->create_publisher<sensor_msgs::msg::Image>("/camera/preview_image", 10);

    RCLCPP_INFO_STREAM(this->get_logger(), "publishers created");

    allocateBuffersMemory();
    RCLCPP_INFO_STREAM(this->get_logger(), "buffer allocated");

    applyCameraParameters();

    int desired_fps = this->declare_parameter<int>("fps", 20);
    int latency = 1000 / desired_fps;
    warnLatency(latency);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(latency),
                                     std::bind(&CameraNode::handleCameraOnTimer, this));
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Cameras timer started with latency of " << latency << "ms");
    param_.save_raw_ = this->declare_parameter<bool>("save_raw_frames_on_timer", false);
    param_.save_converted_ = this->declare_parameter<bool>("save_converted_frames_on_timer", false);
    param_.publish_preview = this->declare_parameter<bool>("publish_preview", false);
    if (param_.save_raw_ || param_.save_converted_) {
        initSnapper();
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "initialisation finished");
}

void CameraNode::warnLatency(int latency) {
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        double d_num = latency;
        CameraGetExposureTime(camera_handles_[i], &d_num);
        if (d_num / 1000 > latency) {
            RCLCPP_WARN_STREAM(this->get_logger(),
                               "Desired FPS is not possible due to exposure of "
                                   << d_num / 1000 << "ms exposure of camera " << i);
        }
    }
}

void CameraNode::measureFPS() {
    rclcpp::Time start_time = this->get_clock()->now();
    const int num_of_frames = 100;
    for (size_t i = 0; i < num_of_frames; ++i) {
        int status = CameraGetImageBuffer(camera_handles_[0], &frame_info_[0], &raw_buffer_[0], 50);
        if (status == CAMERA_STATUS_TIME_OUT) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
        } else if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "ERROR occured in handleCameraOnTimer, error code: " << status);
            abort();
        }
        CameraReleaseImageBuffer(camera_handles_[0], raw_buffer_[0]);
    }
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "measured FPS is: " << num_of_frames / (this->get_clock()->now() - start_time).seconds());
}

CameraNode::~CameraNode() {
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        delete[] converted_buffer_[i];
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "uninit done");
}

void CameraNode::allocateBuffersMemory() {
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        // raw_buffer_[i] = new uint8_t[param_.frame_size_.height * param_.frame_size_.width];
        converted_buffer_[i] =
            new uint8_t[param_.frame_size_.height * param_.frame_size_.width * 3];
    }
}

int CameraNode::getHandle(int i) { return camera_handles_[i]; }

void CameraNode::publishConvertedImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id) {
    int state = CameraImageProcess(
        camera_handles_[camera_id], buffer, converted_buffer_[camera_id], &frame_info_[camera_id]);

    cv::Mat cv_image(
        std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth},
        CV_8UC3,
        converted_buffer_[camera_id]);
    sensor_msgs::msg::Image img_msg;

    cv_bridge::CvImage cv_img(getHeader(timestamp, camera_id), "bgr8", cv_image);
    cv_img.toImageMsg(img_msg);
    signals_.converted_img_pub->publish(img_msg);
    if (param_.publish_preview) {
        cv::resize(cv_img.image, cv_img.image, param_.preview_frame_size_);
        cv_img.toImageMsg(img_msg);
        signals_.small_preview_img_pub->publish(img_msg);
    }
}

void CameraNode::publishRawImage(BYTE *buffer, rclcpp::Time timestamp, int camera_id) {
    cv::Mat cv_image(
        std::vector<int>{frame_info_[camera_id].iHeight, frame_info_[camera_id].iWidth},
        CV_8UC1,
        buffer);
    sensor_msgs::msg::Image img_msg;

    cv_bridge::CvImage cv_img(getHeader(timestamp, camera_id), "8uc1", cv_image);
    cv_img.toImageMsg(img_msg);
    signals_.raw_img_pub->publish(img_msg);
}

void CameraNode::handleCameraOnTimer() {
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        int status =
            CameraGetImageBuffer(camera_handles_[i], &frame_info_[i], &raw_buffer_[i], 100);
        if (status == CAMERA_STATUS_TIME_OUT) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
            return;
        } else if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "ERROR occured in handleCameraOnTimer, error code: " << status);
            abort();
        }
        state_.last_frame_timestamps_[i] = this->get_clock()->now();
    }
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        publishRawImage(raw_buffer_[i], state_.last_frame_timestamps_[i], i);
        publishConvertedImage(raw_buffer_[i], state_.last_frame_timestamps_[i], i);
    }
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        CameraReleaseImageBuffer(camera_handles_[i], raw_buffer_[i]);
    }
}

std_msgs::msg::Header CameraNode::getHeader(rclcpp::Time timestamp, int camera_id) {
    std_msgs::msg::Header header;
    header.stamp = timestamp;
    header.frame_id = "Camera #" + std::to_string(camera_id);

    return header;
}

void CameraNode::initSnapper() {
    int timer_period = this->declare_parameter<int>("timer_period");
    param_.path_to_file_save = this->declare_parameter<std::string>("path_to_save");
    std::chrono::milliseconds time(timer_period * 1000);
    snap_timer_ = this->create_wall_timer(time, std::bind(&CameraNode::saveAllFramesOnTimer, this));

    RCLCPP_INFO_STREAM(this->get_logger(), "snapper initialised");
}

void CameraNode::saveAllFramesOnTimer() {
    if (!state_.save_image_id_) {
        ++state_.save_image_id_;
        return;
    }
    std::string current_path;
    // saving raw as binary
    if (param_.save_raw_) {
        for (int i = 0; i < param_.num_of_cameras_; ++i) {
            current_path = param_.path_to_file_save + "/raw_image_" + std::to_string(i) + '_' +
                           std::to_string(state_.save_image_id_) + ".bin";

            std::ofstream file(current_path, std::ios::binary);
            file.write((char *)raw_buffer_[i],
                       param_.frame_size_.height * param_.frame_size_.width);
            file.close();

            RCLCPP_INFO_STREAM(this->get_logger(), "Image " << current_path << " saved");
        }
    }

    if (param_.save_converted_) {
        // saving converted
        for (int i = 0; i < param_.num_of_cameras_; ++i) {
            current_path = param_.path_to_file_save + "/image_" + std::to_string(i) + '_' +
                           std::to_string(state_.save_image_id_) + ".png";
            cv::Mat cv_image(std::vector<int>{frame_info_[i].iHeight, frame_info_[i].iWidth},
                             CV_8UC3,
                             converted_buffer_[i]);
            cv::imwrite(current_path, cv_image);
            RCLCPP_INFO_STREAM(this->get_logger(), "Image " << current_path << " saved");
        }
    }
    ++state_.save_image_id_;
}

void CameraNode::applyCameraParameters() {
    for (int i = 0; i < param_.num_of_cameras_; ++i) {
        applyParamsToCamera(i);
    }
}

void CameraNode::applyParamsToCamera(int camera_idx) {
    const std::string param_names[] = {"exposure_time",
                                       "contrast",
                                       "gain_rgb",
                                       "gamma",
                                       "saturation",
                                       "sharpness",
                                       "auto_exposure"};
    int handle = camera_handles_[camera_idx];
    std::string prefix = "camera_" + std::to_string(camera_idx + 1) + ".";

    {
        const std::string param = "exposure_time";
        const std::string full_param = prefix + param;
        int status = CameraSetExposureTime(handle, this->declare_parameter<double>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting '" << param << " for camera " << camera_idx
                                            << " failed with status: " << status);
            abort();
        }
        double value;
        CameraGetExposureTime(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter '" << param << "' = " << value);
    }

    {
        const std::string param = "contrast";
        const std::string full_param = prefix + param;
        int status = CameraSetContrast(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting '" << param << " for camera " << camera_idx
                                            << " failed with status: " << status);
            abort();
        }
        int value;
        CameraGetContrast(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter '" << param << "' = " << value);
    }

    {
        const std::string param = "gain_rgb";
        const std::string full_param = prefix + param;
        const std::vector<int64_t> gain = this->declare_parameter<std::vector<int>>(full_param);
        int status = CameraSetGain(handle, gain[0], gain[1], gain[2]);
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting '" << param << " for camera " << camera_idx
                                            << " failed with status: " << status);
            abort();
        }
        int value_r, value_g, value_b;
        CameraGetGain(handle, &value_r, &value_g, &value_b);
        RCLCPP_INFO_STREAM(
            this->get_logger(),
            "Parameter '" << param << "' = " << value_r << ", " << value_g << ", " << value_b);
    }

    {
        const std::string param = "gamma";
        const std::string full_param = prefix + param;
        int status = CameraSetGamma(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting '" << param << " for camera " << camera_idx
                                            << " failed with status: " << status);
            abort();
        }
        int value;
        CameraGetGamma(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter '" << param << "' = " << value);
    }

    {
        const std::string param = "saturation";
        const std::string full_param = prefix + param;
        int status = CameraSetSaturation(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting '" << param << " for camera " << camera_idx
                                            << " failed with status: " << status);
            abort();
        }
        int value;
        CameraGetSaturation(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter '" << param << "' = " << value);
    }

    {
        const std::string param = "sharpness";
        const std::string full_param = prefix + param;
        int status = CameraSetSharpness(handle, this->declare_parameter<int>(full_param));
        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting '" << param << " for camera " << camera_idx
                                            << " failed with status: " << status);
            abort();
        }
        int value;
        CameraGetSharpness(handle, &value);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter '" << param << "' = " << value);
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
            RCLCPP_ERROR_STREAM(this->get_logger(),
                                "Setting parameter " << param << " failed with code: " << status);
            abort();
        }
        RCLCPP_INFO_STREAM(this->get_logger(),
                           "Parameter " << param << " was set to " << auto_exposure);
    }
}