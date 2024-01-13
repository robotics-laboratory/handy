#include "camera.h"
#include "camera_status.h"

#include <cv_bridge/cv_bridge.hpp>

#include <algorithm>

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

CameraNode::CameraNode() : Node("camera_node"), free_bgr_buffer_(QUEUE_CAPACITY) {
    abortIfNot("camera init", CameraSdkInit(0));
    tSdkCameraDevInfo cameras_list[MAX_CAMERA_NUM];
    // param_.camera_num equals to 10 initially to attach all connected cameras
    abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &param_.camera_num));

    RCLCPP_INFO_STREAM(this->get_logger(), "camera number: " << param_.camera_num);

    param_.preview_frame_size.width = this->declare_parameter<int>("preview/width", 640);
    param_.preview_frame_size.height = this->declare_parameter<int>("preview/height", 480);

    RCLCPP_INFO_STREAM(this->get_logger(), "frame preview size=" << param_.preview_frame_size);

    param_.hardware_trigger = this->declare_parameter<bool>("hardware_trigger", false);

    RCLCPP_INFO(this->get_logger(), "hardware trigger mode: %i", param_.hardware_trigger);

    camera_handles_ = std::vector<int>(param_.camera_num);
    frame_sizes_ = std::vector<cv::Size>(param_.camera_num);

    int strobe_polarity = this->declare_parameter<int>("strobe_polarity", 1);
    RCLCPP_INFO_STREAM(
        this->get_logger(), "strobe polarity mode: " << (strobe_polarity == 1) ? "LOW" : "HIGH");
    int strobe_pulse_width = this->declare_parameter<int>("strobe_pulse_width", 500);
    RCLCPP_INFO_STREAM(this->get_logger(), "strobe pulse width = " << strobe_pulse_width);

    const int master_camera_id = [&] {
        std::string str_master_camera_id =
            this->declare_parameter<std::string>("master_camera_id", "1");
        RCLCPP_INFO(this->get_logger(), "master camera id = %s", str_master_camera_id.c_str());

        try {
            return std::stoi(str_master_camera_id);
        } catch (std::exception&) {
            RCLCPP_ERROR(
                this->get_logger(), "invalid master camera id '%s'!", str_master_camera_id.c_str());
            abort();
        }
    }();

    for (int i = 0; i < param_.camera_num; ++i) {
        state_.camera_images[i] = std::make_unique<LockFreeQueue<StampedImagePtr>>(QUEUE_CAPACITY);

        abortIfNot(
            "camera init " + std::to_string(i),
            CameraInit(&cameras_list[i], -1, -1, &camera_handles_[i]));

        int camera_internal_id = getCameraId(camera_handles_[i]);
        RCLCPP_INFO(
            this->get_logger(), "allocated image queue for camera ID = %d", camera_internal_id);

        handle_to_camera_idx[camera_handles_[i]] = i;

        abortIfNot("set icp", i, CameraSetIspOutFormat(camera_handles_[i], CAMERA_MEDIA_TYPE_BGR8));

        auto func = [](CameraHandle idx,
                       BYTE* raw_buffer,
                       tSdkFrameHead* frame_info,
                       PVOID camera_node_instance) -> void {
            reinterpret_cast<CameraNode*>(camera_node_instance)
                ->handleFrame(idx, raw_buffer, frame_info);
        };
        CameraSetCallbackFunction(camera_handles_[i], std::move(func), this, NULL);

        CameraSetTriggerMode(camera_handles_[i], SOFT_TRIGGER);

        if (param_.hardware_trigger && camera_internal_id != master_camera_id) {
            RCLCPP_INFO(this->get_logger(), "treating as slave camera ID = %d", camera_internal_id);
            CameraSetTriggerMode(camera_handles_[i], EXTERNAL_TRIGGER);
            CameraSetExtTrigSignalType(camera_handles_[i], EXT_TRIG_HIGH_LEVEL);
            CameraSetOutPutIOMode(camera_handles_[i], 0, IOMODE_TRIG_INPUT);
        } else {
            RCLCPP_INFO(
                this->get_logger(), "treating as master camera ID = %d", camera_internal_id);
            CameraSetOutPutIOMode(camera_handles_[i], 0, IOMODE_STROBE_OUTPUT);
            CameraSetStrobeMode(camera_handles_[i], STROBE_SYNC_WITH_TRIG_MANUAL);
            CameraSetStrobePolarity(camera_handles_[i], strobe_polarity);
            CameraSetStrobeDelayTime(camera_handles_[i], 0);
            CameraSetStrobePulseWidth(camera_handles_[i], strobe_pulse_width);
        }
        applyParamsToCamera(camera_handles_[i]);
        abortIfNot("start", CameraPlay(camera_handles_[i]));

        if (param_.hardware_trigger && camera_internal_id == master_camera_id) {
            param_.master_camera_idx = i;
            RCLCPP_INFO(
                this->get_logger(),
                "camera ID = %d  idx = %d is saved as master camera",
                camera_internal_id,
                i);
        }
        RCLCPP_INFO(
            this->get_logger(), "inited API and started camera ID = %d", camera_internal_id);
    }
    if (param_.hardware_trigger &&
        param_.master_camera_idx == -1) {  // provided master camera id was not found
        RCLCPP_ERROR(this->get_logger(), "master camera id was not found: %d", master_camera_id);
        abort();
    }
    if (!param_.hardware_trigger) {
        param_.master_camera_idx = 0;
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "all cameras started!");

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
    std::chrono::duration<double> queue_handling_latency(std::max(0.01, (1 / fps) / (fps / 5)));
    RCLCPP_INFO(this->get_logger(), "latency=%fs", param_.latency.count());
    RCLCPP_INFO(this->get_logger(), "queue_latency=%fs", queue_handling_latency.count());
    param_.max_buffer_size = MaxBufSize(frame_sizes_);
    for (int i = 0; i < QUEUE_CAPACITY; ++i) {
        std::shared_ptr<uint8_t[]> new_bgr_buffer(new uint8_t[param_.max_buffer_size]);
        if (!free_bgr_buffer_.push(new_bgr_buffer)) {
            RCLCPP_ERROR(this->get_logger(), "failed to push to queue BGR buffer #%d", i);
            abort();
        }
    }

    call_group_.trigger_timer =
        this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_.camera_soft_trigger = this->create_wall_timer(
        param_.latency, std::bind(&CameraNode::triggerOnTimer, this), call_group_.trigger_timer);

    call_group_.handling_queue_timer =
        this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    timer_.camera_handle_queue_timer = this->create_wall_timer(
        queue_handling_latency,
        std::bind(&CameraNode::handleQueue, this),
        call_group_.handling_queue_timer);
}

CameraNode::~CameraNode() {
    for (int i = 0; i < param_.camera_num; ++i) {
        abortIfNot("camera " + std::to_string(i) + " stop", CameraStop(camera_handles_[i]));
        abortIfNot("camera " + std::to_string(i) + " uninit", CameraUnInit(camera_handles_[i]));
    }
}

void CameraNode::triggerOnTimer() {
    CameraSoftTrigger(camera_handles_[param_.master_camera_idx]);
    if (!param_.hardware_trigger) {
        for (int i = 0; i < param_.camera_num; ++i) {
            if (i == param_.master_camera_idx) {
                continue;
            }
            CameraSoftTrigger(camera_handles_[i]);
        }
    }
}

int CameraNode::getCameraId(int camera_handle) {
    uint8_t camera_id;
    abortIfNot("getting camera id", CameraLoadUserData(camera_handle, 0, &camera_id, 1));
    return static_cast<int>(camera_id);
}

void CameraNode::applyParamsToCamera(int handle) {
    // applying exposure params
    int camera_idx = handle_to_camera_idx[handle];
    std::string prefix = std::to_string(getCameraId(handle)) + ".exposure_params.";
    {
        tSdkCameraCapbility camera_capability;
        abortIfNot("get image size", CameraGetCapability(handle, &camera_capability));
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

    if (!param_.publish_rectified_preview) {
        return;
    }

    prefix = std::to_string(getCameraId(handle)) + ".intrinsic_params.";
    const std::vector<std::string> param_names = {
        "image_size.width", "image_size.height", "camera_matrix", "distorsion_coefs"};
    bool has_all_params = std::all_of(
        param_names.begin(), param_names.end(), [&prefix, this](const std::string& param_name) {
            return this->has_parameter(prefix + param_name);
        });
    if (!has_all_params) {
        RCLCPP_ERROR(this->get_logger(), "camera %d failed to read instrinsic params", camera_idx);
        abort();
    }

    // loading intrinsic parameters
    const int64_t param_image_width = this->declare_parameter<int64_t>(prefix + "image_size.width");
    const int64_t param_image_height =
        this->declare_parameter<int64_t>(prefix + "image_size.height");
    const cv::Size param_image_size(param_image_width, param_image_height);

    const std::vector<double> param_camera_matrix =
        this->declare_parameter<std::vector<double>>(prefix + "camera_matrix");
    const std::vector<double> param_dist_coefs =
        this->declare_parameter<std::vector<double>>(prefix + "distorsion_coefs");

    cameras_intrinsics_.push_back(CameraIntrinsicParameters::loadFromParams(
        param_image_size, param_camera_matrix, param_dist_coefs));
    RCLCPP_INFO(this->get_logger(), "camera=%i loaded intrinsics", camera_idx);
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

void CameraNode::publishBGRImage(
    uint8_t* buffer, rclcpp::Time timestamp, int idx, tSdkFrameHead& frame_inf) {
    const auto header = makeHeader(timestamp, idx);
    std::shared_ptr<uint8_t[]> bgr_buffer;
    while (!free_bgr_buffer_.pop(bgr_buffer)) {
    }

    abortIfNot(
        "get bgr", CameraImageProcess(camera_handles_[idx], buffer, bgr_buffer.get(), &frame_inf));
    cv::Mat image(frame_sizes_[idx], CV_8UC3, bgr_buffer.get());

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
    if (!free_bgr_buffer_.push(bgr_buffer)) {
        RCLCPP_ERROR(this->get_logger(), "unable to push bgr buffer back after use");
        abort();
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

void CameraNode::handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info) {
    const cv::Size size{frame_info->iWidth, frame_info->iHeight};
    if (size != frame_sizes_[handle_to_camera_idx[handle]]) {
        RCLCPP_ERROR_STREAM(
            this->get_logger(),
            "expected frame size " << frame_sizes_[handle] << ", but got " << size);
        abort();
    }
    int frame_size_px = frame_info->iWidth * frame_info->iHeight;
    std::shared_ptr<uint8_t[]> buffer_to_copy(new uint8_t[frame_size_px]);
    std::memcpy(buffer_to_copy.get(), raw_buffer, frame_size_px);
    StampedImagePtr stamped_image_to_add{
        buffer_to_copy, std::move(*frame_info), this->get_clock()->now()};
    if (!state_.camera_images[handle_to_camera_idx[handle]]->push(stamped_image_to_add)) {
        RCLCPP_ERROR(this->get_logger(), "unable to fit into queue! aborting");
        timer_.camera_handle_queue_timer.get()->cancel();
        timer_.camera_soft_trigger.get()->cancel();
        this->~CameraNode();
        abort();
    }
    CameraReleaseImageBuffer(handle, raw_buffer);
}

void CameraNode::handleQueue() {
    for (int i = 0; i < param_.camera_num; ++i) {
        StampedImagePtr stamped_image;
        if (!state_.camera_images[i]->pop(stamped_image)) {
            continue;
        }
        if (param_.publish_raw || param_.publish_raw_preview) {
            publishRawImage(stamped_image.buffer.get(), stamped_image.timestamp, i);
        }
        if (param_.publish_bgr || param_.publish_bgr_preview) {
            publishBGRImage(
                stamped_image.buffer.get(), stamped_image.timestamp, i, stamped_image.frame_info);
        }
    }
}

}  // namespace handy::camera
