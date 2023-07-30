#include "camera.h"

#include <cv_bridge/cv_bridge.hpp>

namespace handy::camera {

using namespace std::chrono_literals;

namespace {

std::string_view toStatusName(int status) {
    switch (status) {
        case CAMERA_STATUS_SUCCESS:
            return "SUCCESS";
        case CAMERA_STATUS_FAILED:
            return "FAILED";
        case CAMERA_STATUS_INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case CAMERA_STATUS_UNKNOW:
            return "UNKNOW";
        case CAMERA_STATUS_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case CAMERA_STATUS_NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case CAMERA_STATUS_PARAMETER_INVALID:
            return "PARAMETER_INVALID";
        case CAMERA_STATUS_PARAMETER_OUT_OF_BOUND:
            return "PARAMETER_OUT_OF_BOUND";
        case CAMERA_STATUS_UNENABLED:
            return "UNENABLED";
        case CAMERA_STATUS_USER_CANCEL:
            return "USER_CANCEL";
        case CAMERA_STATUS_PATH_NOT_FOUND:
            return "PATH_NOT_FOUND";
        case CAMERA_STATUS_SIZE_DISMATCH:
            return "SIZE_DISMATCH";
        case CAMERA_STATUS_TIME_OUT:
            return "TIME_OUT";
        case CAMERA_STATUS_IO_ERROR:
            return "IO_ERROR";
        case CAMERA_STATUS_COMM_ERROR:
            return "COMM_ERROR";
        case CAMERA_STATUS_BUS_ERROR:
            return "BUS_ERROR";
        case CAMERA_STATUS_NO_DEVICE_FOUND:
            return "NO_DEVICE_FOUND";
        case CAMERA_STATUS_NO_LOGIC_DEVICE_FOUND:
            return "NO_LOGIC_DEVICE_FOUND";
        case CAMERA_STATUS_DEVICE_IS_OPENED:
            return "DEVICE_IS_OPENED";
        case CAMERA_STATUS_DEVICE_IS_CLOSED:
            return "DEVICE_IS_CLOSED";
        case CAMERA_STATUS_DEVICE_VEDIO_CLOSED:
            return "DEVICE_VEDIO_CLOSED";
        case CAMERA_STATUS_NO_MEMORY:
            return "NO_MEMORY";
        case CAMERA_STATUS_FILE_CREATE_FAILED:
            return "FILE_CREATE_FAILED";
        case CAMERA_STATUS_FILE_INVALID:
            return "FILE_INVALID";
        case CAMERA_STATUS_WRITE_PROTECTED:
            return "WRITE_PROTECTED";
        case CAMERA_STATUS_GRAB_FAILED:
            return "GRAB_FAILED";
        case CAMERA_STATUS_LOST_DATA:
            return "LOST_DATA";
        case CAMERA_STATUS_EOF_ERROR:
            return "EOF_ERROR";
        case CAMERA_STATUS_BUSY:
            return "BUSY";
        case CAMERA_STATUS_WAIT:
            return "WAIT";
        case CAMERA_STATUS_IN_PROCESS:
            return "IN_PROCESS";
        case CAMERA_STATUS_IIC_ERROR:
            return "IIC_ERROR";
        case CAMERA_STATUS_SPI_ERROR:
            return "SPI_ERROR";
        case CAMERA_STATUS_USB_CONTROL_ERROR:
            return "USB_CONTROL_ERROR";
        case CAMERA_STATUS_USB_BULK_ERROR:
            return "USB_BULK_ERROR";
        case CAMERA_STATUS_SOCKET_INIT_ERROR:
            return "SOCKET_INIT_ERROR";
        case CAMERA_STATUS_GIGE_FILTER_INIT_ERROR:
            return "GIGE_FILTER_INIT_ERROR";
        case CAMERA_STATUS_NET_SEND_ERROR:
            return "NET_SEND_ERROR";
        case CAMERA_STATUS_DEVICE_LOST:
            return "DEVICE_LOST";
        case CAMERA_STATUS_DATA_RECV_LESS:
            return "DATA_RECV_LESS";
        case CAMERA_STATUS_FUNCTION_LOAD_FAILED:
            return "FUNCTION_LOAD_FAILED";
        case CAMERA_STATUS_CRITICAL_FILE_LOST:
            return "CRITICAL_FILE_LOST";
        case CAMERA_STATUS_SENSOR_ID_DISMATCH:
            return "SENSOR_ID_DISMATCH";
        case CAMERA_STATUS_OUT_OF_RANGE:
            return "OUT_OF_RANGE";
        case CAMERA_STATUS_REGISTRY_ERROR:
            return "REGISTRY_ERROR";
        case CAMERA_STATUS_ACCESS_DENY:
            return "ACCESS_DENY";
        case CAMERA_STATUS_CAMERA_NEED_RESET:
            return "CAMERA_NEED_RESET";
        case CAMERA_STATUS_ISP_MOUDLE_NOT_INITIALIZED:
            return "ISP_MOUDLE_NOT_INITIALIZED";
        case CAMERA_STATUS_ISP_DATA_CRC_ERROR:
            return "ISP_DATA_CRC_ERROR";
        case CAMERA_STATUS_MV_TEST_FAILED:
            return "MV_TEST_FAILED";
        case CAMERA_STATUS_INTERNAL_ERR1:
            return "INTERNAL_ERR1";
        case CAMERA_STATUS_U3V_NO_CONTROL_EP:
            return "U3V_NO_CONTROL_EP";
        case CAMERA_STATUS_U3V_CONTROL_ERROR:
            return "U3V_CONTROL_ERROR";
        case CAMERA_STATUS_INVALID_FRIENDLY_NAME:
            return "INVALID_FRIENDLY_NAME";
        case CAMERA_STATUS_FORMAT_ERROR:
            return "FORMAT_ERROR";
        case CAMERA_STATUS_PCIE_OPEN_ERROR:
            return "PCIE_OPEN_ERROR";
        case CAMERA_STATUS_PCIE_COMM_ERROR:
            return "PCIE_COMM_ERROR";
        case CAMERA_STATUS_PCIE_DDR_ERROR:
            return "PCIE_DDR_ERROR";
        default:
            throw std::runtime_error("Unknwown status!");
    }
}

inline int len(const std::string_view& s) { return static_cast<int>(s.size()); }

}  // namespace

void CameraNode::abortIfNot(std::string_view msg, int status) {
    if (status != CAMERA_STATUS_SUCCESS) {
        const auto status_name = toStatusName(status);
        RCLCPP_ERROR(
            this->get_logger(),
            "%.*s, %.*s(%i)",
            len(msg),
            msg.data(),
            len(status_name),
            status_name.data(),
            status);
        abort();
    }
}

void CameraNode::abortIfNot(std::string_view msg, int camera_idx, int status) {
    if (status != CAMERA_STATUS_SUCCESS) {
        const auto status_name = toStatusName(status);
        RCLCPP_ERROR(
            this->get_logger(),
            "%.*s, camera=%i, %.*s(%i)",
            len(msg),
            msg.data(),
            camera_idx,
            len(status_name),
            status_name.data(),
            status);
        abort();
    }
}

CameraNode::CameraNode() : Node("camera_node") {
    abortIfNot("camera init", CameraSdkInit(0));

    int camera_num = 100;  // attach all conntected camers
    tSdkCameraDevInfo cameras_list;
    abortIfNot("camera listing", CameraEnumerateDevice(&cameras_list, &camera_num));

    param_.camera_num = this->declare_parameter<int>("camera_num", 1);
    if (param_.camera_num != camera_num) {
        RCLCPP_ERROR_STREAM(
            this->get_logger(),
            "expected " << param_.camera_num << " cameras, found " << camera_num);
        abort();
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "Camera number: " << param_.camera_num);

    param_.preview_frame_size.width = this->declare_parameter<int>("preview/width", 640);
    param_.preview_frame_size.height = this->declare_parameter<int>("preview/height", 480);

    RCLCPP_INFO_STREAM(this->get_logger(), "frame size=" << param_.frame_size);
    RCLCPP_INFO_STREAM(this->get_logger(), "frame preview size=" << param_.preview_frame_size);

    const int bgr_buffer_size = 3 * param_.frame_size.width * param_.frame_size.height;

    camera_handles_ = std::vector<int>(param_.camera_num);
    raw_buffer_ptr_ = std::vector<uint8_t*>(param_.camera_num);
    bgr_buffer_ = std::make_unique<uint8_t[]>(param_.camera_num * bgr_buffer_size);
    frame_info_ = std::vector<tSdkFrameHead>(param_.camera_num);

    abortIfNot("cameras init", CameraInit(&cameras_list, -1, -1, camera_handles_.data()));

    for (int i = 0; i < param_.camera_num; ++i) {
        abortIfNot("set icp", i, CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8));
        abortIfNot("start", CameraPlay(camera_handles_[i]));
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "all cameras started!");

    param_.publish_raw = this->declare_parameter<bool>("publish_raw", false);
    RCLCPP_INFO(this->get_logger(), "publist raw: %i", param_.publish_raw);

    param_.publish_raw_preview = this->declare_parameter<bool>("publish_raw_preview", false);
    RCLCPP_INFO(this->get_logger(), "publish raw preview: %i", param_.publish_raw_preview);

    param_.publish_bgr = this->declare_parameter<bool>("publish_bgr", false);
    RCLCPP_INFO(this->get_logger(), "publist bgr: %i", param_.publish_bgr);

    param_.publish_bgr_preview = this->declare_parameter<bool>("publish_bgr_preview", false);
    RCLCPP_INFO(this->get_logger(), "publist bgr preview: %i", param_.publish_bgr_preview);

    for (int i = 1; i <= param_.camera_num; ++i) {
        const std::string root = "/camera_" + std::to_string(i);
        constexpr int queue_size = 1;

        if (param_.publish_raw) {
            signals_.raw_img.push_back(this->create_publisher<sensor_msgs::msg::CompressedImage>(
                root + "/raw/image", queue_size));
        }

        if (param_.publish_raw_preview) {
            signals_.raw_preview_img.push_back(
                this->create_publisher<sensor_msgs::msg::CompressedImage>(
                    root + "/raw/preview", queue_size));
        }

        if (param_.publish_bgr) {
            signals_.bgr_img.push_back(this->create_publisher<sensor_msgs::msg::CompressedImage>(
                root + "/bgr/image", queue_size));
        }

        if (param_.publish_bgr_preview) {
            signals_.bgr_preview_img.push_back(
                this->create_publisher<sensor_msgs::msg::CompressedImage>(
                    root + "/bgr/preview", queue_size));
        }
    }

    const auto fps = this->declare_parameter<double>("fps", 20.0);
    param_.latency = std::chrono::duration<double>(1 / fps);
    RCLCPP_INFO(this->get_logger(), "latency=%fs", param_.latency.count());
    timer_ = this->create_wall_timer(param_.latency, std::bind(&CameraNode::handleOnTimer, this));

    applyCameraParameters();
}

void CameraNode::applyCameraParameters() {
    for (int i = 0; i < param_.camera_num; ++i) {
        applyParamsToCamera(i);
    }
}

void CameraNode::applyParamsToCamera(int camera_idx) {
    int handle = camera_handles_[camera_idx];
    std::string prefix = "camera_" + std::to_string(camera_idx) + ".";

    {
        const std::string param = "exposure_time";
        const std::string full_param = prefix + param;
        std::chrono::microseconds exposure(this->declare_parameter<long int>(full_param));

        if (exposure > param_.latency) {
            RCLCPP_INFO(
                this->get_logger(),
                "exposure %lius for camera %i, but latency=%fms",
                exposure.count(),
                camera_idx,
                param_.latency.count());
            abort();
        }

        abortIfNot("set exposure", camera_idx, CameraSetExposureTime(handle, exposure.count()));
        RCLCPP_INFO(this->get_logger(), "camera=%i, exposure=%lius", camera_idx, exposure.count());
    }

    {
        const std::string param = "contrast";
        const std::string full_param = prefix + param;
        const int contrast = this->declare_parameter<int>(full_param);

        abortIfNot("set contrast", camera_idx, CameraSetExposureTime(handle, contrast));
        RCLCPP_INFO(this->get_logger(), "camera=%i, contrast=%i", camera_idx, contrast);
    }

    {
        const std::string param = "analog_gain";
        const std::string full_param = prefix + param;
        const int gain = this->declare_parameter<int>(full_param, -1);

        if (gain != -1) {
            abortIfNot("set analog gain", CameraSetAnalogGain(handle, gain));
            RCLCPP_INFO(this->get_logger(), "camera=%i, analog_gain=%i", camera_idx, gain);
        } else {
            const std::string param = "gain_rgb";
            const std::string full_param = prefix + param;
            const std::vector<long int> gain =
                this->declare_parameter<std::vector<long int>>(full_param);

            if (gain.size() != 3) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "camera=%i, expected gain_rgb as tuple with size 3",
                    camera_idx);
            }

            abortIfNot("set gain", CameraSetGain(handle, gain[0], gain[1], gain[2]));
            RCLCPP_INFO(
                this->get_logger(),
                "camera=%i, gain=[%li, %li, %li]",
                camera_idx,
                gain[0],
                gain[1],
                gain[2]);
        }
    }

    {
        const std::string param = "gamma";
        const std::string full_param = prefix + param;
        const int gamma = this->declare_parameter<int>(full_param);
        abortIfNot("set gamma", CameraSetGamma(handle, gamma));
        RCLCPP_INFO(this->get_logger(), "camera=%i, gamma=%i", camera_idx, gamma);
    }

    {
        const std::string param = "saturation";
        const std::string full_param = prefix + param;
        const int saturation = this->declare_parameter<int>(full_param);
        abortIfNot("set saturation", CameraSetSaturation(handle, saturation));
        RCLCPP_INFO(this->get_logger(), "camera=%i, saturation=%i", camera_idx, saturation);
    }

    {
        const std::string param = "sharpness";
        const std::string full_param = prefix + param;
        const int sharpness = this->declare_parameter<int>(full_param);
        abortIfNot("set sharpness", CameraSetSharpness(handle, sharpness));
        RCLCPP_INFO(this->get_logger(), "camera=%i, sharpness=%i", camera_idx, sharpness);
    }

    {
        const std::string param = "auto_exposure";
        const std::string full_param = prefix + param;
        const bool auto_exposure = this->declare_parameter<bool>(full_param);
        abortIfNot("set auto exposure", CameraSetAeState(handle, auto_exposure));
        RCLCPP_INFO(this->get_logger(), "camera=%i, auto_exposure=%i", camera_idx, auto_exposure);
    }
}

namespace {

std_msgs::msg::Header makeHeader(rclcpp::Time timestamp, int camera_idx) {
    std_msgs::msg::Header header;
    header.stamp = timestamp;
    header.frame_id = "camera_" + std::to_string(camera_idx);
    return header;
}

sensor_msgs::msg::CompressedImage toPngMsg(const cv_bridge::CvImage& cv_image) {
    sensor_msgs::msg::CompressedImage result;
    cv_image.toCompressedImageMsg(result, cv_bridge::Format::PNG);
    return result;
}

sensor_msgs::msg::CompressedImage toJpegMsg(const cv_bridge::CvImage& cv_image) {
    sensor_msgs::msg::CompressedImage result;
    cv_image.toCompressedImageMsg(result, cv_bridge::Format::JPEG);
    return result;
}

cv::Mat rescale(const cv::Mat& image, const cv::Size& size) {
    cv::Mat result;
    cv::resize(image, result, size);
    return result;
}

}  // namespace

void CameraNode::publishBGRImage(uint8_t* buffer, rclcpp::Time timestamp, int idx) {
    const auto header = makeHeader(timestamp, idx);

    const int bgr_buffer_size = 3 * param_.frame_size.width * param_.frame_size.height;
    const auto bgr_buffer = bgr_buffer_.get() + idx * bgr_buffer_size;

    abortIfNot(
        "get bgr", CameraImageProcess(camera_handles_[idx], buffer, bgr_buffer, &frame_info_[idx]));
    cv::Mat image(param_.frame_size, CV_8UC3, bgr_buffer);

    if (param_.publish_bgr) {
        cv_bridge::CvImage cv_image(header, "bgr8", image);
        signals_.bgr_img[idx]->publish(toPngMsg(cv_image));
    }

    if (param_.publish_bgr_preview) {
        cv_bridge::CvImage cv_image(header, "bgr8", rescale(image, param_.preview_frame_size));
        signals_.bgr_preview_img[idx]->publish(toJpegMsg(cv_image));
    }
}

void CameraNode::publishRawImage(uint8_t* buffer, rclcpp::Time timestamp, int camera_idx) {
    const auto header = makeHeader(timestamp, camera_idx);
    cv::Mat image(param_.frame_size, CV_8UC1, buffer);

    if (param_.publish_raw) {
        cv_bridge::CvImage cv_image(header, "mono8", image);
        signals_.raw_img[camera_idx]->publish(toPngMsg(cv_image));
    }

    if (param_.publish_raw_preview) {
        cv_bridge::CvImage cv_image(header, "mono8", rescale(image, param_.preview_frame_size));
        signals_.raw_preview_img[camera_idx]->publish(toJpegMsg(cv_image));
    }
}

void CameraNode::handleOnTimer() {
    const auto now = this->get_clock()->now();

    for (int i = 0; i < param_.camera_num; ++i) {
        const auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(param_.latency);
        const int status = CameraGetImageBuffer(
            camera_handles_[i], &frame_info_[i], &raw_buffer_ptr_[i], timeout.count());

        if (status != CAMERA_STATUS_SUCCESS) {
            const auto status_name = toStatusName(status);
            RCLCPP_WARN(
                this->get_logger(),
                "miss frame camera=%i, %.*s(%i)",
                i,
                len(status_name),
                status_name.data(),
                status);
            continue;
        }

        const cv::Size size{frame_info_[i].iWidth, frame_info_[i].iHeight};

        if (size != param_.frame_size) {
            RCLCPP_ERROR_STREAM(
                this->get_logger(),
                "expected frame size " << param_.frame_size << ", but got " << size);
            abort();
        }
    }

    for (int i = 0; i < param_.camera_num; ++i) {
        if (param_.publish_raw || param_.publish_raw_preview) {
            publishRawImage(raw_buffer_ptr_[i], now, i);
        }

        if (param_.publish_bgr || param_.publish_bgr_preview) {
            publishBGRImage(raw_buffer_ptr_[i], now, i);
        }

        CameraReleaseImageBuffer(camera_handles_[i], raw_buffer_ptr_[i]);
    }
}

}  // namespace handy::camera
