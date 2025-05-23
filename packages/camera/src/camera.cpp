#include "camera.h"
#include "camera_status.h"

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

namespace {

template<typename T>
T getUnsignedDifference(T& a, T& b) {
    return (a > b) ? (a - b) : (b - a);
}

// TODO add timestamp checks?
bool checkForSynchronization(
    std::vector<std::vector<handy::camera::StampedImageBuffer>>& camera_images,
    std::vector<size_t>& indices) {
    int first_frame_id = camera_images[0][indices[0]].frame_id;
    for (size_t i = 1; i < indices.size(); ++i) {
        if (first_frame_id != camera_images[i][indices[i]].frame_id) {
            return false;
        }
    }
    return true;
}

}  // namespace

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

MappedFileManager::MappedFileManager() {
    std::cout << "MAX_SIZE INIT " << kMmapLargeConstant << '\n';
    printf("MAX_SIZE INIT %ld\n", kMmapLargeConstant);
    int64_t page_size = sysconf(_SC_PAGE_SIZE);
    kMmapLargeConstant = kMmapLargeConstant / page_size * page_size + page_size;
}

void MappedFileManager::init(CameraRecorder* recorder_instance, const std::string& filepath) {
    recorder_instance_ = recorder_instance;

    file_ = open(filepath.c_str(), O_RDWR | O_CREAT, S_IWUSR | S_IRUSR);
    if (!file_) {
        perror("file open");
    }
    printf("MAX_SIZE INIT %ld\n", kMmapLargeConstant);
    if (ftruncate(file_, kMmapLargeConstant)) {
        perror("file resize");
    }

    internal_mapping_size_ = kMmapLargeConstant;
    printf("MAX_SIZE INIT %ld\n", kMmapLargeConstant);

    mmaped_ptr_ = mmap(nullptr, kMmapLargeConstant, PROT_READ | PROT_WRITE, MAP_SHARED, file_, 0);
    if (mmaped_ptr_ == MAP_FAILED) {
        printf("errorno=%d\n", errno);
        perror("mmap");
        exit(EXIT_FAILURE);
    }
    printf("allocated on %p\n", mmaped_ptr_);
    std::vector<uint8_t> tmp_data(10, 0);
    std::memcpy(mmaped_ptr_, static_cast<void*>(tmp_data.data()), 10);
    printf("memcpy from %p on %d successful\n", mmaped_ptr_, 10);
}

void MappedFileManager::write(const std::byte* data, uint64_t size) { handleWrite(data, size); }

void MappedFileManager::handleWrite(const std::byte* data, uint64_t size) {
    while (busy_writing.exchange(true)) {
    }

    if (this->size() + size - internal_mapping_start_offset_ > kMmapLargeConstant) {
        // recorder_instance_->param_.save_to_file = false;
        doubleMappingSize();
        // recorder_instance_->param_.save_to_file = true;
    }

    uint8_t* current_data_ptr =
        static_cast<uint8_t*>(mmaped_ptr_) + this->size() - internal_mapping_start_offset_;

    std::memcpy(static_cast<void*>(current_data_ptr), reinterpret_cast<const void*>(data), size);
    size_ += size;

    busy_writing = false;
}

void MappedFileManager::doubleMappingSize() {
    // MS_ASYNC -- flag to schedule syncronizing to the disk and not wait
    // for its completion
    if (msync(mmaped_ptr_, kMmapLargeConstant, MS_ASYNC) == -1) {
        perror("msync");
        exit(1);
    }
    if (munmap(mmaped_ptr_, kMmapLargeConstant) == -1) {
        perror("munmap");
        exit(1);
    }

    int64_t page_size = sysconf(_SC_PAGE_SIZE);
    internal_mapping_start_offset_ = size_ / page_size * page_size;
    uint64_t new_size = internal_mapping_size_ * 2;
    uint64_t aligned_size = ((new_size + 1024 * 1024 + page_size - 1) / page_size) * page_size;

    if (ftruncate(file_, aligned_size) == -1) {
        perror("ftruncate");
        exit(1);
    }

    void* ptr = mmap(
        nullptr,
        kMmapLargeConstant,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        file_,
        internal_mapping_start_offset_);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    mmaped_ptr_ = ptr;
    internal_mapping_size_ = aligned_size;
}

void MappedFileManager::end() {
    // MS_SYNC -- flag to sync and wait. No rush because if exiting
    if (msync(mmaped_ptr_, kMmapLargeConstant, MS_SYNC) == -1) {
        perror("Could not sync the file to disk");
        exit(EXIT_FAILURE);
    }
    // free the mmapped memory
    if (munmap(mmaped_ptr_, kMmapLargeConstant) == -1) {
        perror("Error un-mmapping the file");
        exit(EXIT_FAILURE);
    }
    // shrink to the real content size
    int64_t page_size = sysconf(_SC_PAGE_SIZE);
    size_t aligned_fact_size = size_ / page_size * page_size + page_size;
    if (ftruncate(file_, aligned_fact_size) == -1) {
        perror("error resizing the file");
        exit(EXIT_FAILURE);
    }
    close(file_);
}

uint64_t MappedFileManager::size() const { return size_; }

CameraRecorder::CameraRecorder(
    const char* param_file, const char* output_filename, bool save_to_file) {
    param_.param_file = param_file;
    param_.output_filename = output_filename;
    param_.save_to_file = save_to_file;

    if (save_to_file) {
        // if needed register saveCallbackLauncher as a callback when images are syncronized
        mcap::McapWriterOptions options("");
        options.noChunking = true;

        options.noMessageIndex = false;
        options.noAttachmentCRC = true;
        options.noAttachmentIndex = true;
        options.noMetadataIndex = true;
        options.noStatistics = true;

        std::string filename_str(output_filename);
        file_manager_.init(this, filename_str);
        mcap_writer_.open(file_manager_, options);
    }

    abortIfNot("camera init", CameraSdkInit(0));
    tSdkCameraDevInfo cameras_list[10];
    abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &state_.camera_num));

    // init all collections for the number of attached cameras
    state_.frame_sizes.resize(state_.camera_num);
    state_.camera_handles.resize(state_.camera_num);
    state_.frame_ids = std::vector<std::atomic<int>>(state_.camera_num);
    state_.current_buffer_idx = std::vector<std::atomic<size_t>>(state_.camera_num);

    // read common params from launch file
    YAML::Node launch_params = YAML::LoadFile(param_.param_file)["parameters"];
    const auto fps = launch_params["fps"].as<int>();
    param_.master_camera_id = launch_params["master_camera_id"].as<int>();
    param_.use_hardware_triger = launch_params["hardware_triger"].as<bool>();
    param_.latency = std::chrono::duration<double>(1. / fps);
    printf("latency=%fs\n", param_.latency.count());

    // set up each camera
    printf("camera number: %d\n", state_.camera_num);
    for (int i = 0; i < state_.camera_num; ++i) {
        state_.camera_images[i] =
            std::make_unique<boost::lockfree::queue<StampedImageBuffer>>(kQueueCapacity);
        state_.free_buffers[i] = std::make_unique<boost::lockfree::queue<uint8_t*>>(kQueueCapacity);
        state_.free_buffer_cnts[i] = 0;

        mcap::Schema schema("raw_bayer_scheme", "", "");
        mcap_writer_.addSchema(schema);
        state_.bayer_schema_id = schema.id;

        mcap::Channel channel(
            "/camera_" + std::to_string(i + 1) + "/raw/image",
            "raw_bayer_encoding",
            state_.bayer_schema_id);
        mcap_writer_.addChannel(channel);
        state_.mcap_channels_ids_.push_back(channel.id);

        abortIfNot(
            "camera init " + std::to_string(i),
            CameraInit(&cameras_list[i], -1, -1, &state_.camera_handles[i]));
        state_.handle_to_idx[state_.camera_handles[i]] = i;

        abortIfNot(
            "set icp", i, CameraSetIspOutFormat(state_.camera_handles[i], CAMERA_MEDIA_TYPE_BGR8));

        std::string path_to_file = output_filename;
        path_to_file += std::to_string(i);

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
    }

    // determine the largest frame_size, allocate pool and push all free buffers
    Size max_frame_size = *std::max_element(
        state_.frame_sizes.begin(), state_.frame_sizes.begin(), [](Size& first, Size& second) {
            return first.area() < second.area();
        });

    // TODO: fix signedness
    state_.buffers =
        CameraPool(max_frame_size.height, max_frame_size.width, kQueueCapacity * kMaxCameraNum);
    printf("%d pools initialised\n", kQueueCapacity * kMaxCameraNum);

    // init queues and push pointers to buffers
    for (int i = 0; i < state_.camera_num; ++i) {
        for (int j = 0; j < kQueueCapacity; ++j) {
            state_.free_buffers[i]->push(state_.buffers.getRawFrame(i * kQueueCapacity + j));
            state_.free_buffer_cnts[i]++;
        }
    }

    for (int i = 0; i < state_.camera_num; ++i) {
        abortIfNot("reset timestamp", i, CameraRstTimeStamp(state_.camera_handles[i]));
    }
    for (int i = 0; i < state_.camera_num; ++i) {
        abortIfNot("start", CameraPlay(state_.camera_handles[i]));
        abortIfNot("reset timestamp", i, CameraRstTimeStamp(state_.camera_handles[i]));
        printf("inited API and started camera handle = %d\n", state_.camera_handles[i]);
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

    for (int i = 0; i < state_.camera_num; ++i) {
        abortIfNot("camera " + std::to_string(i) + " stop", CameraStop(state_.camera_handles[i]));
        abortIfNot(
            "camera " + std::to_string(i) + " uninit", CameraUnInit(state_.camera_handles[i]));
    }
    mcap_writer_.close();
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
    std::vector<std::vector<StampedImageBuffer>> camera_images(state_.camera_num);
    while (state_.running) {
        std::unique_lock<std::mutex> lock(state_.synchronizer_mutex);
        state_.synchronizer_condvar.wait_for(lock, std::chrono::milliseconds(100));

        // empty all queues
        for (int i = 0; i < state_.camera_num; ++i) {
            camera_images[i].emplace_back();
            while (state_.camera_images[i]->pop(camera_images[i].back())) {
                camera_images[i].emplace_back();
            }
            // delete the last element
            camera_images[i].erase(--camera_images[i].end());
        }
        // if all cameras have at least one frame
        if (std::any_of(
                camera_images.begin(),
                camera_images.end(),
                [&](std::vector<StampedImageBuffer>& vec) { return vec.empty(); })) {
            continue;
        }

        // balance pools and free impossible to sync buffers
        size_t max_size = camera_images[0].size();
        size_t min_size = camera_images[0].size();
        for (const std::vector<StampedImageBuffer>& v : camera_images) {
            max_size = std::max(max_size, v.size());
            min_size = std::min(min_size, v.size());
        }
        // if imbalance is significant (let us say 10)
        if (max_size - min_size > 10) {
            for (size_t i = 0; i < camera_images.size(); ++i) {
                while (camera_images[i].size() > min_size) {
                    auto front_elem_to_erase = camera_images[i].begin();
                    while (!this->state_.free_buffers[i]->push(front_elem_to_erase->raw_buffer)) {
                    }
                    state_.free_buffer_cnts[i]++;
                    camera_images[i].erase(front_elem_to_erase);
                }
            }
        }

        // prepare iterations
        std::vector<size_t> indices(camera_images.size(), 0);
        std::vector<size_t> sizes(camera_images.size());

        std::transform(
            camera_images.begin(),
            camera_images.end(),
            sizes.begin(),
            [](const std::vector<StampedImageBuffer>& v) { return v.size(); });

        // next vector of indices
        auto next = [&]() {
            for (size_t i = 0; i < indices.size(); ++i) {
                if (++indices[i] < sizes[i]) {
                    return true;
                }
                indices[i] = 0;
            }
            return false;
        };

        // make single structure as a shared ptr
        // deleter is expected to push buffers back to the queue and then free the structure
        auto custom_deleter = [this](SynchronizedFrameBuffers* sync_buffers_ptr) {
            for (size_t i = 0; i < sync_buffers_ptr->images.size(); ++i) {
                if (!sync_buffers_ptr->images[i]) {
                    continue;
                }
                while (!this->state_.free_buffers[i]->push(sync_buffers_ptr->images[i])) {
                }
                state_.free_buffer_cnts[i]++;
            }
            delete sync_buffers_ptr;
        };
        std::shared_ptr<SynchronizedFrameBuffers> sync_buffers(
            new SynchronizedFrameBuffers, custom_deleter);
        sync_buffers->images.resize(state_.camera_num, nullptr);
        sync_buffers->image_sizes.resize(state_.camera_num);

        // iterate through all possible combinations and check for equal ids and timestamps
        do {
            if (!checkForSynchronization(camera_images, indices)) {
                continue;
            }
            // casting new structure
            sync_buffers->timestamp = camera_images[0][indices[0]].timestamp;
            for (size_t i = 0; i < indices.size(); ++i) {
                StampedImageBuffer& current_buffer = camera_images[i][indices[i]];
                sync_buffers->images[i] = current_buffer.raw_buffer;
                sync_buffers->image_sizes[i] = state_.frame_sizes[i];

                camera_images[i].erase(camera_images[i].begin() + indices[i]);
                --sizes[i];
            }
            // invoke saving buffers in a detached thread as other callbacks?
            if (param_.save_to_file) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                std::thread(&CameraRecorder::saveSynchronizedBuffers, this, sync_buffers).detach();
            }
            // invoking all subscribers in detached threads
            for (size_t i = 0; i < state_.registered_callbacks.size(); ++i) {
                std::thread(state_.registered_callbacks[i], sync_buffers).detach();
            }

            // creating new structure
            sync_buffers = std::shared_ptr<SynchronizedFrameBuffers>(
                new SynchronizedFrameBuffers,
                custom_deleter);  // custom deleter adds buffer pointer back to the queue
            sync_buffers->images.resize(state_.camera_num, nullptr);
            sync_buffers->image_sizes.resize(state_.camera_num);
        } while (next());
    }
}

void CameraRecorder::handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info) {
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

    uint8_t* free_buffer;
    int cnt = 0;
    while (!state_.free_buffers[state_.handle_to_idx[handle]]->pop(free_buffer)) {
        int value_to_print = state_.free_buffer_cnts[state_.handle_to_idx[handle]];
        printf(
            "cant pop free buffer camera_idx=%d remaining_ptrs=%d\n",
            state_.handle_to_idx[handle],
            value_to_print);
        cnt++;
        if (cnt > 2) {
            state_.running = false;
            sleep(1);
            exit(EXIT_FAILURE);
        }
    }
    state_.free_buffer_cnts[state_.handle_to_idx[handle]]--;

    std::memcpy(free_buffer, raw_buffer, frame_size_px);

    int frame_id = state_.frame_ids[state_.handle_to_idx[handle]].fetch_add(1);
    StampedImageBuffer stamped_buffer_to_add{
        free_buffer,  // raw buffer
        *frame_info,
        state_.handle_to_idx[handle],
        frame_id,
        frame_info->uiTimeStamp};
    if (!state_.camera_images[state_.handle_to_idx[handle]]->push(stamped_buffer_to_add)) {
        printf("unable to fit into queue! exiting\n");
        exit(EXIT_FAILURE);
    }
    state_.synchronizer_condvar.notify_one();

    CameraReleaseImageBuffer(handle, raw_buffer);
}

// no lint to insist on copying shared_ptr and incrementing ref counter
// NOLINTNEXTLINE
void CameraRecorder::saveSynchronizedBuffers(std::shared_ptr<SynchronizedFrameBuffers> images) {
    auto start = std::chrono::high_resolution_clock::now();
    mcap::Timestamp timestamp(static_cast<uint64_t>(images->timestamp) * 100000ul);
    for (int i = 0; i < state_.camera_num; ++i) {
        mcap::Message msg{
            state_.mcap_channels_ids_[i],
            0,
            timestamp,
            timestamp,
            state_.frame_sizes[i].area(),
            reinterpret_cast<const std::byte*>(images->images[i])};
        mcap::Status status = mcap_writer_.write(msg);
        if (!status.ok()) {
            state_.running = false;
            sleep(1);
            exit(EXIT_FAILURE);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    if (elapsed.count() > 13) {
        std::cout << "WARNING: Save function took more than 13 ms " << elapsed.count() << '\n';
    }
}

void CameraRecorder::registerSubscriberCallback(const CameraSubscriberCallback& callback) {
    state_.registered_callbacks.push_back(callback);
}

int CameraRecorder::getCameraId(int camera_handle) {
    uint8_t camera_id;
    abortIfNot("getting camera id", CameraLoadUserData(camera_handle, 0, &camera_id, 1));
    return static_cast<int>(camera_id);
}

void CameraRecorder::stopInstance() { state_.running = false; }

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
