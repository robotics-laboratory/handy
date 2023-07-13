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
        RCLCPP_ERROR_STREAM(
            this->get_logger(), "ERROR occured during cameras initialisation, code: " << status);
    }

    for (const int handle : camera_handles_) {
        CameraPlay(handle);
        CameraSetIspOutFormat(handle, CAMERA_MEDIA_TYPE_BGR8);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "cameras started");

    signals_.raw_img_pub = this->create_publisher<sensor_msgs::msg::Image>("/camera/raw_image", 10);
    signals_.converted_img_pub =
        this->create_publisher<sensor_msgs::msg::Image>("/camera/converted_image", 10);
    signals_.small_preview_img_pub =
        this->create_publisher<sensor_msgs::msg::Image>("/camera/preview_image", 10);

    frame_size_ = cv::Size(1280, 1024);
    preview_frame_size_ = cv::Size(640, 480);

    RCLCPP_INFO_STREAM(this->get_logger(), "publishers created");
    allocateBuffersMemory();
    RCLCPP_INFO_STREAM(this->get_logger(), "buffer allocated");


    applyCameraParameters();
    
    RCLCPP_INFO_STREAM(this->get_logger(), "start at");
    

    for (size_t i = 0; i < 100; ++i) {
        rclcpp::Time start_time = this->get_clock()->now();
        int status =
            CameraGetImageBuffer(camera_handles_[0], &frame_info_[0], &raw_buffer_[0], 50);
        //RCLCPP_INFO_STREAM(this->get_logger(), (this->get_clock()->now() - start_time).seconds());
        if (status == CAMERA_STATUS_TIME_OUT) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
        } else if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(), "ERROR occured in handleCameraOnTimer, error code: " << status);
        }
        CameraReleaseImageBuffer(camera_handles_[0], raw_buffer_[0]);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "stop at");

    timer_ = this->create_wall_timer(500ms, std::bind(&CameraNode::handleCameraOnTimer, this));

    RCLCPP_INFO_STREAM(this->get_logger(), "initialisation finished");


}

CameraNode::~CameraNode() {
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        free(converted_buffer_[i]);
        // free(raw_buffer_[i]);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "uninit done");
}

void CameraNode::allocateBuffersMemory() {
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        raw_buffer_[i] = (unsigned char *)malloc(frame_size_.height * frame_size_.width);
        converted_buffer_[i] = (unsigned char *)malloc(frame_size_.height * frame_size_.width * 3);
    }
}

int CameraNode::getHandle(int i) { return camera_handles_[i]; }

void CameraNode::publishConvertedImage(
    BYTE *buffer, rclcpp::Time timestamp, int camera_id, bool publish_preview = false) {
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
    if (publish_preview) {
        cv::resize(cv_img.image, cv_img.image, preview_frame_size_);
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
    for (size_t i = 0; i < num_of_cameras_; ++i) {
        int status =
            CameraGetImageBuffer(camera_handles_[i], &frame_info_[i], &raw_buffer_[i], 100);
        if (status == CAMERA_STATUS_TIME_OUT) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "ERROR: timeout, waiting for raw buffer");
            return;
        } else if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(), "ERROR occured in handleCameraOnTimer, error code: " << status);
            return;
        }
        last_frame_timestamps_[i] = this->get_clock()->now();
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
    header.frame_id = std::to_string(camera_id);

    return header;
}

void CameraNode::applyCameraParameters() {
    const std::string param_names[] = {
        "exposure time",
        "contrast",
        "gain_R",
        "gain_G",
        "gain_B",
        "gamma",
        "saturation",
        "sharpness",
        "auto_exposure"};
    std::string current_param;
    for (int i = 0; i < num_of_cameras_; ++i) {
        for (const std::string &param : param_names) {
            current_param = param + '_' + std::to_string(i + 1);
            applyParamsToCamera(camera_handles_[i], param, current_param);
        }
    }
    // RCLCPP_INFO_STREAM(this->get_logger(), "exposure: " << params["exposure"]);
}

void CameraNode::applyParamsToCamera(
    int camera_handle, const std::string &param_type, std::string &param_name) {
    int status;
    int i_num;
    double d_num;
    if (param_type == "exposure time") {
        status = CameraSetExposureTime(camera_handle, this->declare_parameter<double>(param_name));

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetExposureTime(camera_handle, &d_num);
        RCLCPP_INFO_STREAM(
            this->get_logger(), "Parameter " << param_name << " was set to " << d_num);

    } else if (param_type == "contrast") {
        status = CameraSetContrast(camera_handle, this->declare_parameter<int>(param_name));

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetContrast(camera_handle, &i_num);
        RCLCPP_INFO_STREAM(
            this->get_logger(), "Parameter " << param_name << " was set to " << i_num);

    } else if (param_type == "gain_R") {
        int r, g, b;
        CameraGetGain(camera_handle, &r, &g, &b);
        status = CameraSetGain(camera_handle, this->declare_parameter<int>(param_name), g, b);

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetGain(camera_handle, &r, &g, &b);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter " << param_name << " was set to " << r);

    } else if (param_type == "gain_G") {
        int r, g, b;
        CameraGetGain(camera_handle, &r, &g, &b);
        status = CameraSetGain(camera_handle, r, this->declare_parameter<int>(param_name), b);

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetGain(camera_handle, &r, &g, &b);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter " << param_name << " was set to " << g);

    } else if (param_type == "gain_B") {
        int r, g, b;
        CameraGetGain(camera_handle, &r, &g, &b);
        status = CameraSetGain(camera_handle, r, g, this->declare_parameter<int>(param_name));

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetGain(camera_handle, &r, &g, &b);
        RCLCPP_INFO_STREAM(this->get_logger(), "Parameter " << param_name << " was set to " << b);
    } else if (param_type == "gamma") {
        status = CameraSetGamma(camera_handle, this->declare_parameter<int>(param_name));

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetGamma(camera_handle, &i_num);
        RCLCPP_INFO_STREAM(
            this->get_logger(), "Parameter " << param_name << " was set to " << i_num);

    } else if (param_type == "saturation") {
        status = CameraSetSaturation(camera_handle, this->declare_parameter<int>(param_name));

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetSaturation(camera_handle, &i_num);
        RCLCPP_INFO_STREAM(
            this->get_logger(), "Parameter " << param_name << " was set to " << i_num);

    } else if (param_type == "sharpness") {
        status = CameraSetSharpness(camera_handle, this->declare_parameter<int>(param_name));

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting arameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetSharpness(camera_handle, &i_num);
        RCLCPP_INFO_STREAM(
            this->get_logger(), "Parameter " << param_name << " was set to " << i_num);

    } else if (param_type == "auto_exposure") {
        bool auto_exposure = this->declare_parameter<bool>(param_name);
        if (auto_exposure) {
            status = CameraSetAeThreshold(camera_handle, 10);
        } else {
            status = CameraSetAeThreshold(camera_handle, 1000);
        }

        if (status != CAMERA_STATUS_SUCCESS) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "Setting parameter " << param_name << " failed with code: " << status);
            return;
        }
        CameraGetSharpness(camera_handle, &i_num);
        RCLCPP_INFO_STREAM(
            this->get_logger(), "Parameter " << param_name << " was set to " << i_num);

    } else {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Unknown parameter: " << param_name);
        return;
    }
}