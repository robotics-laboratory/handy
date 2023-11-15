#include "camera.h"
#include "camera_status.h"

#include <cv_bridge/cv_bridge.hpp>

namespace handy::camera {

using namespace std::chrono_literals;

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

namespace {
int maxBufSize(const std::vector<cv::Size>& frame_sizes) {
    int max_size = 0;
    for (size_t i = 0; i < frame_sizes.size(); ++i) {
        max_size = std::max(max_size, 3 * frame_sizes[i].width * frame_sizes[i].height);
    }
    return max_size;
}
}  // namespace

void CameraNode::cameraCallback(
    CameraHandle idx, BYTE* raw_buffer, tSdkFrameHead* frame_info, PVOID camera_node_instance) {
    reinterpret_cast<CameraNode*>(camera_node_instance)->handleFrame(idx, raw_buffer, frame_info);
}

CameraNode::CameraNode() : Node("camera_node") {
    abortIfNot("camera init", CameraSdkInit(0));

    int camera_num = 100;  // attach all conntected cameras
    tSdkCameraDevInfo cameras_list[100];
    abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &camera_num));

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

    RCLCPP_INFO_STREAM(this->get_logger(), "frame preview size=" << param_.preview_frame_size);

    camera_handles_ = std::vector<int>(param_.camera_num);
    raw_buffer_ptr_ = std::vector<uint8_t*>(param_.camera_num);
    frame_info_ = std::vector<tSdkFrameHead>(param_.camera_num);
    frame_sizes_ = std::vector<cv::Size>(param_.camera_num);

    for (int i = 0; i < param_.camera_num; ++i) {
        abortIfNot(
            "camera init " + std::to_string(i),
            CameraInit(&cameras_list[i], -1, -1, &camera_handles_[i]));
        camera_idxs[camera_handles_[i]] = i;
        abortIfNot("set icp", i, CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8));

        void* placeholder_1;
        CAMERA_SNAP_PROC* placeholder_2;
        auto func = [](CameraHandle idx,
                       BYTE* raw_buffer,
                       tSdkFrameHead* frame_info,
                       PVOID camera_node_instance) -> void {
            reinterpret_cast<CameraNode*>(camera_node_instance)
                ->handleFrame(idx, raw_buffer, frame_info);
        };
        CameraSetCallbackFunction(camera_handles_[i], std::move(func), this, placeholder_2);

        CameraSetTriggerMode(camera_handles_[i], (i == 0) ? SOFT_TRIGGER : EXTERNAL_TRIGGER);
        if (i != 0) {
            CameraSetExtTrigSignalType(camera_handles_[i], EXT_TRIG_HIGH_LEVEL);
            CameraSetOutPutIOMode(camera_handles_[i], 0, IOMODE_TRIG_INPUT);
        }
        CameraSetOutPutIOMode(camera_handles_[i], 0, IOMODE_STROBE_OUTPUT);
        CameraSetStrobeMode(camera_handles_[i], STROBE_SYNC_WITH_TRIG_MANUAL);
        CameraSetStrobePolarity(camera_handles_[i], 1);
        CameraSetStrobeDelayTime(camera_handles_[i], 0);
        CameraSetStrobePulseWidth(camera_handles_[i], 500);

        abortIfNot("start", CameraPlay(camera_handles_[i]));
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "all cameras started!");

    param_.calibration_file_path = this->declare_parameter<std::string>(
        "calibration_file_path", "param_save/camera_params.yaml");
    RCLCPP_INFO(
        this->get_logger(),
        "parameters will be read from: %s",
        param_.calibration_file_path.c_str());

    param_.publish_raw = this->declare_parameter<bool>("publish_raw", false);
    RCLCPP_INFO(this->get_logger(), "publish raw: %i", param_.publish_raw);

    param_.publish_raw_preview = this->declare_parameter<bool>("publish_raw_preview", false);
    RCLCPP_INFO(this->get_logger(), "publish raw preview: %i", param_.publish_raw_preview);

    param_.publish_bgr = this->declare_parameter<bool>("publish_bgr", false);
    RCLCPP_INFO(this->get_logger(), "publish bgr: %i", param_.publish_bgr);

    param_.publish_bgr_preview = this->declare_parameter<bool>("publish_bgr_preview", false);
    RCLCPP_INFO(this->get_logger(), "publish bgr preview: %i", param_.publish_bgr_preview);

    param_.publish_rectified_preview =
        this->declare_parameter<bool>("publish_rectified_preview", false);
    RCLCPP_INFO(
        this->get_logger(),
        "publish rectified preview image: %i",
        param_.publish_rectified_preview);

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

        if (param_.publish_rectified_preview) {
            signals_.rectified_preview_img.push_back(
                this->create_publisher<sensor_msgs::msg::CompressedImage>(
                    root + "/rectified/preview", queue_size));
        }
    }

    const auto fps = this->declare_parameter<double>("fps", 20.0);
    param_.latency = std::chrono::duration<double>(1 / fps);
    RCLCPP_INFO(this->get_logger(), "latency=%fs", param_.latency.count());

    timer_.camera_soft_trigger = this->create_wall_timer(
        param_.latency, [this]() -> void { CameraSoftTrigger(camera_handles_[0]); });

    applyCameraParameters();

    param_.max_buffer_size = maxBufSize(frame_sizes_);
    bgr_buffer_ = std::make_unique<uint8_t[]>(param_.camera_num * param_.max_buffer_size);

    if (param_.publish_rectified_preview) {
        initCalibParams();
    }
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
        tSdkCameraCapbility camera_capability;
        abortIfNot(
            "get image size", CameraGetCapability(camera_handles_[camera_idx], &camera_capability));
        tSdkImageResolution* resolution_data = camera_capability.pImageSizeDesc;
        RCLCPP_INFO(
            this->get_logger(),
            "camera=%i, image_size=(%i, %i)",
            camera_idx,
            resolution_data->iWidth,
            resolution_data->iHeight);
        frame_sizes_[camera_idx] = cv::Size(resolution_data->iWidth, resolution_data->iHeight);
    }

    {
        const std::string param = "auto_exposure";
        const std::string full_param = prefix + param;
        const bool auto_exposure = this->declare_parameter<bool>(full_param);
        abortIfNot("set auto exposure", CameraSetAeState(handle, auto_exposure));
        RCLCPP_INFO(this->get_logger(), "camera=%i, auto_exposure=%i", camera_idx, auto_exposure);
    }

    {
        const std::string param = "exposure_time";
        const std::string full_param = prefix + param;
        std::chrono::microseconds exposure(this->declare_parameter<int64_t>(full_param));

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

        abortIfNot("set contrast", camera_idx, CameraSetContrast(handle, contrast));
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
            const std::vector<int64_t> gain =
                this->declare_parameter<std::vector<int64_t>>(full_param);

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
}

namespace {

std_msgs::msg::Header makeHeader(const rclcpp::Time& timestamp, int camera_idx) {
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

void CameraNode::publishBGRImage(uint8_t* buffer, const rclcpp::Time& timestamp, int idx) {
    const auto header = makeHeader(timestamp, idx);
    auto* const bgr_buffer = bgr_buffer_.get() + idx * param_.max_buffer_size;

    abortIfNot(
        "get bgr", CameraImageProcess(camera_handles_[idx], buffer, bgr_buffer, &frame_info_[idx]));
    cv::Mat image(frame_sizes_[idx], CV_8UC3, bgr_buffer);

    if (param_.publish_bgr) {
        cv_bridge::CvImage cv_image(header, "bgr8", image);
        signals_.bgr_img[idx]->publish(toPngMsg(cv_image));
    }

    if (param_.publish_bgr_preview) {
        cv_bridge::CvImage cv_image(header, "bgr8", rescale(image, param_.preview_frame_size));
        signals_.bgr_preview_img[idx]->publish(toJpegMsg(cv_image));
    }

    if (param_.publish_rectified_preview) {
        cv::Mat undistorted_image = cameras_intrinsics_[idx].undistortImage(image);
        cv_bridge::CvImage cv_image(
            header, "bgr8", rescale(undistorted_image, param_.preview_frame_size));
        signals_.rectified_preview_img[idx]->publish(toJpegMsg(cv_image));
    }
}

void CameraNode::publishRawImage(uint8_t* buffer, const rclcpp::Time& timestamp, int camera_idx) {
    const auto header = makeHeader(timestamp, camera_idx);
    cv::Mat image(frame_sizes_[camera_idx], CV_8UC1, buffer);

    if (param_.publish_raw) {
        cv_bridge::CvImage cv_image(header, "mono8", image);
        signals_.raw_img[camera_idx]->publish(toPngMsg(cv_image));
    }

    if (param_.publish_raw_preview) {
        cv_bridge::CvImage cv_image(header, "mono8", rescale(image, param_.preview_frame_size));
        signals_.raw_preview_img[camera_idx]->publish(toJpegMsg(cv_image));
    }
}

void CameraNode::handleFrame(CameraHandle idx, BYTE* raw_buffer, tSdkFrameHead* frame_info) {
    const auto now = this->get_clock()->now();

    const cv::Size size{frame_info->iWidth, frame_info->iHeight};

    RCLCPP_INFO(this->get_logger(), "called camera #%i callback", idx);

    if (size != frame_sizes_[camera_idxs[idx]]) {
        RCLCPP_ERROR_STREAM(
            this->get_logger(),
            "expected frame size " << frame_sizes_[idx] << ", but got " << size);
        abort();
    }

    if (param_.publish_raw || param_.publish_raw_preview) {
        publishRawImage(raw_buffer, now, camera_idxs[idx]);
    }

    if (param_.publish_bgr || param_.publish_bgr_preview) {
        publishBGRImage(raw_buffer, now, camera_idxs[idx]);
    }

    // CameraReleaseImageBuffer(camera_handles_[i], raw_buffer_ptr_[i]);
}

void CameraNode::initCalibParams() {
    for (int idx = 0; idx < param_.camera_num; ++idx) {
        RCLCPP_INFO_STREAM(this->get_logger(), "loading camera " << idx << " parameters");
        std::string calibration_name = "camera_" + std::to_string(idx);
        cameras_intrinsics_.push_back(
            CameraIntrinsicParameters::loadFromYaml(param_.calibration_file_path));
    }
}
}  // namespace handy::camera
