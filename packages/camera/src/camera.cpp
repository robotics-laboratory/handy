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
#include <cstring>
#include <sys/stat.h>
#include <sys/mman.h>

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
    static constexpr int BUFFER_CAPACITY = 1000;

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
        state_.current_buffer_idx = std::vector<std::atomic<size_t>>(state_.camera_num);
        state_.alligned_buffers = std::vector<void*>(state_.camera_num);

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

            // if node is launch in soft trigger mode
            CameraSetTriggerMode(state_.camera_handles[i], SOFT_TRIGGER);

            CameraSetCallbackFunction(state_.camera_handles[i], std::move(func), this, nullptr);

            applyParamsToCamera(state_.camera_handles[i]);
            abortIfNot("start", CameraPlay(state_.camera_handles[i]));
            printf("inited API and started camera handle = %d\n", state_.camera_handles[i]);
        }

        const auto fps = YAML::LoadFile(param_.param_file)["parameters"]["fps"].as<int>();
        param_.latency = std::chrono::duration<double>(1. / fps);
        printf("latency=%fs\n", param_.latency.count());
        printf("frames will be taken %d\n", fps * duration);

        for (int trigger_cnt = 0; trigger_cnt < BUFFER_CAPACITY; ++trigger_cnt) {
            for (int i = 0; i < state_.camera_num; ++i) {
                CameraSoftTrigger(state_.camera_handles[i]);
            }
            std::this_thread::sleep_for(param_.latency);
            // if ((trigger_cnt + 1) % BUFFER_CAPACITY == 0) {
            //     printf("starting writing\n");

            //     for (int i = 0; i < state_.camera_num; ++i) {
            //         std::lock_guard<std::mutex> lock(state_.file_mutexes[i]);
            //         write(
            //             state_.files[i],
            //             state_.alligned_buffers[i],
            //             state_.frame_sizes[i].area() * BUFFER_CAPACITY);
            //         state_.current_buffer_idx[i].store(0);
            //     }
            //     break;
            // }
        }

        std::this_thread::sleep_for(param_.latency * 100);

        printf("%d %d\n", state_.counters[0].load(), state_.counters[1].load());

        for (int i = 0; i < state_.camera_num; ++i) {
            abortIfNot(
                "camera " + std::to_string(i) + " stop", CameraStop(state_.camera_handles[i]));
            abortIfNot(
                "camera " + std::to_string(i) + " uninit", CameraUnInit(state_.camera_handles[i]));

            long page_size = sysconf(_SC_PAGE_SIZE);
            long size =
                state_.frame_sizes[i].area() * BUFFER_CAPACITY / page_size * page_size + page_size;
            if (msync(state_.alligned_buffers[i], size, MS_SYNC) == -1) {
                perror("Could not sync the file to disk");
            }
            // Free the mmapped memory
            if (munmap(state_.alligned_buffers[i], size) == -1) {
                close(state_.files[i]);
                perror("Error un-mmapping the file");
                exit(1);
            }

            close(state_.files[i]);
            // free(state_.alligned_buffers[i]);
        }
    }

  private:
    void handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info) {
        auto start = std::chrono::high_resolution_clock::now();

        const int camera_idx = state_.handle_to_idx[handle];
        ++state_.counters[camera_idx];
        std::lock_guard<std::mutex> lock(state_.file_mutexes[camera_idx]);
        size_t buffer_idx = state_.current_buffer_idx[camera_idx].fetch_add(1);
        printf("%d buffer id %ld\n", handle, buffer_idx);

        std::memcpy(
            (uint8_t*)state_.alligned_buffers[camera_idx] +
                state_.frame_sizes[camera_idx].area() * buffer_idx,
            raw_buffer,
            frame_info->iWidth * frame_info->iHeight);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        // std::cout << "Function took " << elapsed.count() << " ms to complete.\n";

        if (elapsed.count() > 10) {
            std::cout << "WARNING: Function took more than 10 ms " << elapsed.count() << '\n';
        }
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

            // struct statx res;
            // int status =
            //     statx(state_.files[camera_idx], "", AT_EMPTY_PATH, STATX_BASIC_STATS, &res);
            // if (status == -1) {
            //     printf("statx error\n");
            //     exit(EXIT_FAILURE);
            // }
            // printf("allignment: %d\n", res.stx_blksize);
            // status = posix_memalign(
            //     &state_.alligned_buffers[camera_idx],
            //     res.stx_blksize,
            //     state_.frame_sizes[camera_idx].area() * BUFFER_CAPACITY);
            // if (status != 0) {
            //     printf("posix_memalign error %d\n", status);
            //     exit(EXIT_FAILURE);
            // }

            const std::string path_to_file = param_.output_filename + "_" + camera_id_str + ".out";
            state_.files[camera_idx] =
                open(path_to_file.c_str(), O_RDWR | O_CREAT, S_IWUSR | S_IRUSR);
            if (state_.files[camera_idx] == -1) {
                printf("failed to open %s\n", camera_id_str.c_str());
                exit(EXIT_FAILURE);
            }
            long page_size = sysconf(_SC_PAGE_SIZE);

            if (ftruncate(
                    state_.files[camera_idx],
                    state_.frame_sizes[camera_idx].area() * BUFFER_CAPACITY / page_size *
                            page_size +
                        page_size) == -1) {
                close(state_.files[camera_idx]);
                perror("Error resizing the file");
                exit(1);
            }

            state_.alligned_buffers[camera_idx] = mmap(
                NULL,
                state_.frame_sizes[camera_idx].area() * BUFFER_CAPACITY / page_size * page_size +
                    page_size,
                PROT_WRITE,
                MAP_SHARED,
                state_.files[camera_idx],
                0);
            // malloc(state_.frame_sizes[camera_idx].area() * BUFFER_CAPACITY);
            if (state_.alligned_buffers[camera_idx] == MAP_FAILED) {
                perror("mmap");
                printf("malloc error\n");
                exit(EXIT_FAILURE);
            }
            // int status = posix_memalign(
            //     &state_.alligned_buffers[camera_idx],
            //     res.stx_blksize,
            //     state_.frame_sizes[camera_idx].area() * BUFFER_CAPACITY);
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
        size_t area() const { return static_cast<size_t>(width * height); };

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
        std::vector<std::atomic<size_t>> current_buffer_idx;
        int camera_num = 2;
        std::vector<int> files;
        std::map<int, int> handle_to_idx;
        std::vector<Size> frame_sizes;
        std::vector<std::mutex> file_mutexes;
        std::vector<int> camera_handles;
        std::vector<void*> alligned_buffers;
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
