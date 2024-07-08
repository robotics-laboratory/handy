#include "camera_status.h"
#include "camera.h"

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <sys/mman.h>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

namespace handy::camera {

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

CameraRecorder::CameraRecorder(const char* param_file, const char* output_filename) {
    param_.param_file = {param_file};
    param_.output_filename = {output_filename};

    abortIfNot("camera init", CameraSdkInit(0));
    tSdkCameraDevInfo cameras_list[10];
    abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &state_.camera_num));

    // init all collections for the number of attached cameras
    state_.file_mutexes = std::vector<std::mutex>(state_.camera_num);
    state_.frame_sizes.resize(state_.camera_num);
    state_.files.resize(state_.camera_num);
    state_.camera_handles.resize(state_.camera_num);
    state_.counters = std::vector<std::atomic<int>>(state_.camera_num);
    state_.current_buffer_idx = std::vector<std::atomic<size_t>>(state_.camera_num);
    state_.alligned_buffers = std::vector<void*>(state_.camera_num);

    // read common params from launch file
    YAML::Node launch_params = YAML::LoadFile(param_.param_file)["parameters"];
    const auto fps = launch_params["fps"].as<int>();
    const auto duration = launch_params["duration"].as<int>();
    param_.master_camera_id = launch_params["master_camera_id"].as<int>();
    param_.use_hardware_triger = launch_params["hardware_triger"].as<bool>();
    param_.latency = std::chrono::duration<double>(1. / fps);
    param_.frames_to_take = fps * duration;
    printf("latency=%fs\n", param_.latency.count());
    printf("frames will be taken %d\n", param_.frames_to_take);

    // set up each camera
    printf("camera number: %d\n", state_.camera_num);
    for (int i = 0; i < state_.camera_num; ++i) {
        abortIfNot(
            "camera init " + std::to_string(i),
            CameraInit(&cameras_list[i], -1, -1, &state_.camera_handles[i]));
        state_.handle_to_idx[state_.camera_handles[i]] = i;

        abortIfNot(
            "set icp", i, CameraSetIspOutFormat(state_.camera_handles[i], CAMERA_MEDIA_TYPE_BGR8));

        // if node is launch in soft trigger mode
        if (!param_.use_hardware_triger || state_.camera_handles[i] == param_.master_camera_id) {
            CameraSetTriggerMode(state_.camera_handles[i], SOFT_TRIGGER);
        } else {
            CameraSetTriggerMode(state_.camera_handles[i], EXTERNAL_TRIGGER);
        }

        auto func = [](CameraHandle idx,
                       BYTE* raw_buffer,
                       tSdkFrameHead* frame_info,
                       PVOID camera_node_instance) -> void {
            reinterpret_cast<CameraRecorder*>(camera_node_instance)
                ->handleFrame(idx, raw_buffer, frame_info);
        };
        CameraSetCallbackFunction(state_.camera_handles[i], std::move(func), this, nullptr);

        applyParamsToCamera(state_.camera_handles[i]);
        abortIfNot("start", CameraPlay(state_.camera_handles[i]));
        printf("inited API and started camera handle = %d\n", state_.camera_handles[i]);
    }

    // determine the largest frame_size, allocate pool and push all free buffers
    Size max_frame_size = *std::max_element(
        state_.frame_sizes.begin(), state_.frame_sizes.begin(), [](Size& first, Size& second) {
            return first.area() < second.area();
        });

    state_.buffers =
        CameraPool(max_frame_size.height, max_frame_size.width, kQueueCapacity * kMaxCameraNum);
    printf("%d pools initialised", kQueueCapacity * kMaxCameraNum);

    // init queues and push pointers to buffers
    for (int i = 0; i < state_.camera_num; ++i) {
        for (size_t j = 0; j < kQueueCapacity; ++j) {
            state_.free_buffers[i]->push(
                {state_.buffers.getRawFrame(i * kQueueCapacity + j),
                 state_.buffers.getBgrFrame(i * kQueueCapacity + j)});
        }
    }

    // start trigger in a separate thread
    state_.threads.emplace_back(&CameraRecorder::triggerCamera, this);
    // start queue handler in a separate thread
    state_.threads.emplace_back(&CameraRecorder::synchronizeQueues, this);
}

CameraRecorder::~CameraRecorder() {
    state_.running = false;
    // waiting for all threads to stop by flag state_.running
    for (auto iter = state_.threads.begin(); iter != state_.threads.end(); ++iter) {
        iter->join();
    }

    printf(
        "Got and written from cameras: %d %d\n",
        state_.counters[0].load(),
        state_.counters[1].load());

    for (int i = 0; i < state_.camera_num; ++i) {
        abortIfNot("camera " + std::to_string(i) + " stop", CameraStop(state_.camera_handles[i]));
        abortIfNot(
            "camera " + std::to_string(i) + " uninit", CameraUnInit(state_.camera_handles[i]));

        int64_t page_size = sysconf(_SC_PAGE_SIZE);
        int64_t size = state_.frame_sizes[i].area() * param_.frames_to_take / page_size * page_size
                       + page_size;
        if (msync(state_.alligned_buffers[i], size, MS_SYNC) == -1) {
            perror("Could not sync the file to disk");
        }
        // free the mmapped memory
        if (munmap(state_.alligned_buffers[i], size) == -1) {
            close(state_.files[i]);
            perror("Error un-mmapping the file");
            exit(1);
        }

        close(state_.files[i]);
    }
}

void CameraRecorder::triggerCamera() {
    boost::asio::io_service io;
    boost::posix_time::milliseconds interval(
        static_cast<int>(param_.latency.count() * 1000));  // `latency` milliseconds
    boost::asio::deadline_timer timer(io, interval);

    while (state_.running) {
        timer.wait();
        for (int i = 0; i < state_.camera_num; ++i) {
            if (!param_.use_hardware_triger
                || state_.camera_handles[i] == param_.master_camera_id) {
                CameraSoftTrigger(state_.camera_handles[i]);
            }
        }
        timer.expires_at(timer.expires_at() + interval);
    }
}

void CameraRecorder::synchronizeQueues() {
    while (state_.running) {
        
    }
}

void CameraRecorder::handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info) {
    // TODO: delete elapsed time?
    auto start = std::chrono::high_resolution_clock::now();

    Size size{frame_info->iWidth, frame_info->iHeight};
    if (size != state_.frame_sizes[state_.handle_to_idx[handle]]) {
        printf(
            "expected frame size (%d; %d), but got (%d, %d)",
            state_.frame_sizes[handle].width,
            state_.frame_sizes[handle].height,
            size.width,
            size.height);
        exit(EXIT_FAILURE);
    }
    int frame_size_px = frame_info->iWidth * frame_info->iHeight;

    BufferPair free_buffers;
    while (!state_.free_buffers[state_.handle_to_idx[handle]]->pop(free_buffers)) {
    }
    std::memcpy(free_buffers.first, raw_buffer, frame_size_px);
    StampedImageBuffer stamped_buffer_to_add{
        free_buffers.first,   // raw buffer
        free_buffers.second,  // bgr buffer
        *frame_info,
        state_.handle_to_idx[handle],
        frame_info->uiTimeStamp};
    if (!state_.camera_images[state_.handle_to_idx[handle]]->push(stamped_buffer_to_add)) {
        printf("unable to fit into queue! exiting");
        exit(EXIT_FAILURE);
    }
    CameraReleaseImageBuffer(handle, raw_buffer);

    // TODO: delete elapsed time?
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    if (elapsed.count() > 10) {
        std::cout << "WARNING: Function took more than 10 ms " << elapsed.count() << '\n';
    }
}

int CameraRecorder::getCameraId(int camera_handle) {
    uint8_t camera_id;
    abortIfNot("getting camera id", CameraLoadUserData(camera_handle, 0, &camera_id, 1));
    return static_cast<int>(camera_id);
}

void CameraRecorder::applyParamsToCamera(int handle) {
    const int camera_idx = state_.handle_to_idx[handle];
    const std::string camera_id_str = std::to_string(getCameraId(handle));
    const YAML::Node camera_params = YAML::LoadFile(param_.param_file)["parameters"][camera_id_str];

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

        // const std::string path_to_file = param_.output_filename + "_" + camera_id_str + ".out";
        // state_.files[camera_idx] = open(path_to_file.c_str(), O_RDWR | O_CREAT, S_IWUSR |
        // S_IRUSR); if (state_.files[camera_idx] == -1) {
        //     printf("failed to open %s\n", camera_id_str.c_str());
        //     exit(EXIT_FAILURE);
        // }
        // int64_t page_size = sysconf(_SC_PAGE_SIZE);

        // if (ftruncate(
        //         state_.files[camera_idx],
        //         state_.frame_sizes[camera_idx].area() * param_.frames_to_take / page_size
        //                 * page_size
        //             + page_size)
        //     == -1) {
        //     close(state_.files[camera_idx]);
        //     perror("Error resizing the file");
        //     exit(1);
        // }

        // state_.alligned_buffers[camera_idx] = mmap(
        //     nullptr,
        //     state_.frame_sizes[camera_idx].area() * param_.frames_to_take / page_size * page_size
        //         + page_size,
        //     PROT_WRITE,
        //     MAP_SHARED,
        //     state_.files[camera_idx],
        //     0);
        // if (state_.alligned_buffers[camera_idx] == MAP_FAILED) {
        //     perror("mmap");
        //     exit(EXIT_FAILURE);
        // }
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
        const auto contrast = camera_params[param].as<int>();

        abortIfNot("set contrast", camera_idx, CameraSetContrast(handle, contrast));
        printf("camera=%i, contrast=%i\n", camera_idx, contrast);
    }

    {
        const std::string param = "analog_gain";
        const auto gain = camera_params[param].as<int>();

        if (gain != -1) {
            abortIfNot("set analog gain", CameraSetAnalogGain(handle, gain));
            printf("camera=%i, analog_gain=%i\n", camera_idx, gain);
        } else {
            const std::string param = "gain_rgb";
            const auto gain = camera_params[param].as<std::vector<int64_t>>();

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
}  // namespace handy::camera

// ./camera_bin <param_file> <output_filename>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("help\n./camera_bin <param_file> <output_filename>\n");
        return 0;
    }
    handy::camera::CameraRecorder writer(argv[1], argv[2]);
    printf("finished writing to file %s\n", argv[2]);

    return 0;
}
