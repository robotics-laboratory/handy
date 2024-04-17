#include "camera_status.h"
#include <mutex>
#include "CameraApi.h"
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <ctime>
#include <atomic>
#include <iostream>

#include <yaml-cpp/yaml.h>
#include <algorithm>

using namespace std::chrono_literals;

void abortIfNot(std::string_view msg, int status) {
    if (status != CAMERA_STATUS_SUCCESS) {
        const auto status_name = toStatusName(status);
        printf(
            "%.*s, %.*s(%i)\n", len(msg), msg.data(), len(status_name), status_name.data(), status);
        exit(EXIT_FAILURE);
    }
}

void abortIfNot(std::string_view msg, int camera_idx, int status) {
    if (status != CAMERA_STATUS_SUCCESS) {
        const auto status_name = toStatusName(status);
        printf(
            "%.*s, camera=%i, %.*s(%i)\n",
            len(msg),
            msg.data(),
            camera_idx,
            len(status_name),
            status_name.data(),
            status);
        exit(EXIT_FAILURE);
    }
}

class Writer {
  public:
    Writer(char* param_file, char* output_filename, int duration) {
        param_.param_file = {param_file};
        param_.output_filename = {output_filename};
        param_.duration = duration;

        abortIfNot("camera init", CameraSdkInit(0));
        tSdkCameraDevInfo cameras_list[10];
        abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &state_.camera_num));

        state_.file_mutexes = std::vector<std::mutex>(state_.camera_num);
        state_.frame_sizes.resize(state_.camera_num);
        state_.files.resize(state_.camera_num);
        state_.camera_handles.resize(state_.camera_num);
        state_.counters = std::vector<std::atomic<int>>(state_.camera_num);

        printf("camera number: %d\n", state_.camera_num);

        for (int i = 0; i < state_.camera_num; ++i) {
            abortIfNot(
                "camera init " + std::to_string(i),
                CameraInit(&cameras_list[i], -1, -1, &state_.camera_handles[i]));
            state_.handle_to_idx[state_.camera_handles[i]] = i;

            abortIfNot(
                "set icp",
                i,
                CameraSetIspOutFormat(state_.camera_handles[i], CAMERA_MEDIA_TYPE_BGR8));

            auto func = [](CameraHandle idx,
                           BYTE* raw_buffer,
                           tSdkFrameHead* frame_info,
                           PVOID camera_node_instance) -> void {
                reinterpret_cast<Writer*>(camera_node_instance)
                    ->handleFrame(idx, raw_buffer, frame_info);
            };

            CameraSetCallbackFunction(state_.camera_handles[i], std::move(func), this, nullptr);

            // if node is launch in soft trigger mode
            CameraSetTriggerMode(state_.camera_handles[i], SOFT_TRIGGER);

            applyParamsToCamera(state_.camera_handles[i]);
            abortIfNot("start", CameraPlay(state_.camera_handles[i]));
            printf("inited API and started camera handle = %d\n", state_.camera_handles[i]);

            const std::string path_to_file = "camera_" + std::to_string(i) + ".out";
            state_.files[i] = open(path_to_file.c_str(), O_WRONLY | O_CREAT, S_IWUSR | S_IRUSR);
            if (state_.files[i] == -1) {
                printf("failed to open %d\n", i);
                exit(EXIT_FAILURE);
            }
        }

        const auto fps = YAML::LoadFile(param_.param_file)["parameters"]["fps"].as<int>();
        param_.latency = std::chrono::duration<double>(1. / fps);
        printf("latency=%fs\n", param_.latency.count());
        printf("frames will be taken %d\n", fps * duration);

        for (int trigger_cnt = 0; trigger_cnt < fps * duration; ++trigger_cnt) {
            for (int i = 0; i < state_.camera_num; ++i) {
                CameraSoftTrigger(state_.camera_handles[i]);
            }
            std::this_thread::sleep_for(param_.latency);
        }
        std::this_thread::sleep_for(param_.latency * 10);

        printf("%d %d\n", state_.counters[0].load(), state_.counters[1].load());

        for (int i = 0; i < state_.camera_num; ++i) {
            abortIfNot(
                "camera " + std::to_string(i) + " stop", CameraStop(state_.camera_handles[i]));
            abortIfNot(
                "camera " + std::to_string(i) + " uninit", CameraUnInit(state_.camera_handles[i]));

            close(state_.files[i]);
        }
    }

  private:
    void handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info) {
        auto start = std::chrono::system_clock::now();

        const int camera_idx = state_.handle_to_idx[handle];
        ++state_.counters[camera_idx];
        std::lock_guard<std::mutex> lock(state_.file_mutexes[camera_idx]);
        write(state_.files[camera_idx], raw_buffer, frame_info->iWidth * frame_info->iHeight);
        CameraReleaseImageBuffer(handle, raw_buffer);
        // Some computation here
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);

        std::cout << "elapsed: " << elapsed_seconds.count() << "s" << std::endl;
    }

    int getCameraId(int camera_handle) {
        uint8_t camera_id;
        abortIfNot("getting camera id", CameraLoadUserData(camera_handle, 0, &camera_id, 1));
        return static_cast<int>(camera_id);
    }

    void applyParamsToCamera(int handle) {
        const int camera_idx = state_.handle_to_idx[handle];
        const std::string camera_id_str = std::to_string(getCameraId(handle));
        const YAML::Node camera_params =
            YAML::LoadFile(param_.param_file)["parameters"][camera_id_str];

        // applying exposure params
        {
            tSdkCameraCapbility camera_capability;
            abortIfNot("get image size", CameraGetCapability(handle, &camera_capability));
            tSdkImageResolution* resolution_data = camera_capability.pImageSizeDesc;
            printf(
                "camera=%i, image_size=(%i, %i)\n",
                camera_idx,
                resolution_data->iWidth,
                resolution_data->iHeight);
            state_.frame_sizes[camera_idx] = {resolution_data->iWidth, resolution_data->iHeight};
        }

        {
            const std::string param = "auto_exposure";
            const bool auto_exposure = camera_params[param].as<bool>();
            abortIfNot("set auto exposure", CameraSetAeState(handle, auto_exposure));
            printf("camera=%i, auto_exposure=%i\n", camera_idx, auto_exposure);
        }

        {
            const std::string param = "exposure_time";
            std::chrono::microseconds exposure(camera_params[param].as<int64_t>());

            if (exposure > param_.latency) {
                printf(
                    "exposure %lius for camera %i, but latency=%fms\n",
                    exposure.count(),
                    camera_idx,
                    param_.latency.count());
                exit(EXIT_FAILURE);
            }

            abortIfNot("set exposure", camera_idx, CameraSetExposureTime(handle, exposure.count()));
            printf("camera=%i, exposure=%lius\n", camera_idx, exposure.count());
        }

        {
            const std::string param = "contrast";
            const int contrast = camera_params[param].as<int>();

            abortIfNot("set contrast", camera_idx, CameraSetContrast(handle, contrast));
            printf("camera=%i, contrast=%i\n", camera_idx, contrast);
        }

        {
            const std::string param = "analog_gain";
            const int gain = camera_params[param].as<int>();

            if (gain != -1) {
                abortIfNot("set analog gain", CameraSetAnalogGain(handle, gain));
                printf("camera=%i, analog_gain=%i\n", camera_idx, gain);
            } else {
                const std::string param = "gain_rgb";
                const std::vector<int64_t> gain = camera_params[param].as<std::vector<int64_t>>();

                if (gain.size() != 3) {
                    printf("camera=%i, expected gain_rgb as tuple with size 3\n", camera_idx);
                    exit(EXIT_FAILURE);
                }

                abortIfNot("set gain", CameraSetGain(handle, gain[0], gain[1], gain[2]));
                printf("camera=%i, gain=[%li, %li, %li]\n", camera_idx, gain[0], gain[1], gain[2]);
            }
        }

        {
            const std::string param = "gamma";
            const int gamma = camera_params[param].as<int>();
            abortIfNot("set gamma", CameraSetGamma(handle, gamma));
            printf("camera=%i, gamma=%i\n", camera_idx, gamma);
        }

        {
            const std::string param = "saturation";
            const int saturation = camera_params[param].as<int>();
            abortIfNot("set saturation", CameraSetSaturation(handle, saturation));
            printf("camera=%i, saturation=%i\n", camera_idx, saturation);
        }

        {
            const std::string param = "sharpness";
            const int sharpness = camera_params[param].as<int>();
            abortIfNot("set sharpness", CameraSetSharpness(handle, sharpness));
            printf("camera=%i, sharpness=%i\n", camera_idx, sharpness);
        }
    }

    struct Size {
        int area() const { return width * height; };

        int width;
        int height;
    };

    struct Params {
        int duration = 0;                             // in seconds
        std::chrono::duration<double> latency{50.0};  // in milliseconds
        std::string param_file;
        std::string output_filename;
        int fps = 20;
    } param_{};

    struct State {
        std::vector<std::atomic<int>> counters;
        int camera_num = 2;
        std::vector<int> files;
        std::map<int, int> handle_to_idx;
        std::vector<Size> frame_sizes;
        std::vector<std::mutex> file_mutexes;
        std::vector<int> camera_handles;
    } state_{};
};

// ./camera_bin <param_file> <output_filename> <duration>
int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("./camera_bin <param_file> <output_filename> <duration>\n");
        return 0;
    }
    int duration = [&] {
        std::string str_master_camera_id(argv[3]);
        try {
            return std::stoi(str_master_camera_id);
        } catch (std::exception&) {
            printf("invalid master camera id '%s'!\n", str_master_camera_id.c_str());
            exit(EXIT_FAILURE);
        }
    }();

    Writer writer(argv[1], argv[2], duration);
    printf("finished writing to file %s\n", argv[2]);

    return 0;
}
