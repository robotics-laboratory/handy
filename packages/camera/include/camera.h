#include "CameraApi.h"

#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdint>

#include <boost/lockfree/queue.hpp>
#include "mcap_vendor/mcap/writer.hpp"

using namespace std::chrono_literals;

namespace handy::camera {

class CameraRecorder;

class MappedFileManager final : public mcap::IWritable {
  public:
    MappedFileManager();
    ~MappedFileManager() override{};

    void init(CameraRecorder* recorder_instance, const std::string& filepath);
    void write(const std::byte* data, uint64_t size);
    void handleWrite(const std::byte* data, uint64_t size) override;
    void doubleMappingSize();
    void end() override;
    uint64_t size() const override;

    uint64_t kMmapLargeConstant = 1ULL * 1024 * 1024 * 1024;  // 2 GB

  private:
    int file_ = 0;
    void* mmaped_ptr_ = nullptr;
    CameraRecorder* recorder_instance_ = nullptr;  // to be able to call stopInstance()
    uint64_t size_ = 0;
    uint64_t internal_mapping_size_ = 0;
    uint64_t internal_mapping_start_offset_ = 0;
    std::atomic<bool> busy_writing = false;
};

struct Size {
    size_t area() const { return static_cast<size_t>(width * height); };

    int width = 0;
    int height = 0;

    bool operator==(const Size& other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const Size& other) const { return !(*this == other); }
};

struct StampedImageBuffer {
    uint8_t* raw_buffer = nullptr;
    tSdkFrameHead frame_info{};
    int camera_idx = -1;
    int frame_id = -1;
    uint32_t timestamp = 0;
};

struct SynchronizedFrameBuffers {
    std::vector<uint8_t*> images;
    std::vector<Size> image_sizes;
    uint32_t timestamp = 0;  // in 0.1 milliseconds
};

struct CameraPool {
  public:
    CameraPool() = default;
    CameraPool(size_t height, size_t width, size_t frame_n) :
        frame_n_(frame_n), raw_frame_size_(height * width), bgr_frame_size_(height * width * 3) {
        raw_.resize(raw_frame_size_ * frame_n_);
        bgr_.resize(bgr_frame_size_ * frame_n_);
    }

    uint8_t* getRawFrame(size_t frame_idx) { return raw_.data() + frame_idx * raw_frame_size_; }
    uint8_t* getBgrFrame(size_t frame_idx) { return bgr_.data() + frame_idx * bgr_frame_size_; }

  private:
    size_t frame_n_ = 0;
    size_t raw_frame_size_ = 0;
    size_t bgr_frame_size_ = 0;
    std::vector<uint8_t> raw_ = {};
    std::vector<uint8_t> bgr_ = {};
};

class CameraRecorder {
  public:
    // CameraSubscriberCallback is required to be lock-free and finish within param_.latency
    // millisecond
    using CameraSubscriberCallback = std::function<void(std::shared_ptr<SynchronizedFrameBuffers>)>;

    CameraRecorder(const char* param_file, const char* output_filename, bool save_to_file = false);
    ~CameraRecorder();
    void registerSubscriberCallback(const CameraSubscriberCallback& callback);
    void stopInstance();

    constexpr static int kMaxCameraNum = 2;
    constexpr static int kQueueCapacity = 80;

  private:
    void handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info);
    void triggerCamera();
    void synchronizeQueues();
    static int getCameraId(int camera_handle);
    void applyParamsToCamera(int handle);
    // no lint to insist on copying shared_ptr and incrementing ref counter
    // NOLINTNEXTLINE
    void saveSynchronizedBuffers(std::shared_ptr<SynchronizedFrameBuffers> images);

    struct Params {
        std::chrono::duration<double> latency{50.0};  // in milliseconds
        std::string param_file;
        std::string output_filename;
        int master_camera_id = 1;
        bool use_hardware_triger = false;
        std::atomic<bool> save_to_file = false;
    } param_{};

    struct State {
        std::array<std::unique_ptr<boost::lockfree::queue<StampedImageBuffer>>, kMaxCameraNum>
            camera_images;
        std::array<std::unique_ptr<boost::lockfree::queue<uint8_t*>>, kMaxCameraNum> free_buffers;
        std::array<std::atomic<int>, kMaxCameraNum> free_buffer_cnts;

        std::atomic<bool> running = true;
        std::vector<std::thread> threads;  // trigger thread, queue handler thread
        std::vector<std::atomic<int>> frame_ids;
        std::vector<std::atomic<size_t>> current_buffer_idx;
        int camera_num = 2;
        std::map<int, int> handle_to_idx;
        std::vector<Size> frame_sizes;
        std::vector<int> camera_handles;
        std::vector<CameraSubscriberCallback> registered_callbacks;
        CameraPool buffers;
        std::vector<mcap::ChannelId> mcap_channels_ids_;
        mcap::SchemaId bayer_schema_id;

        std::condition_variable synchronizer_condvar;
        std::mutex synchronizer_mutex;
    } state_{};

    mcap::McapWriter mcap_writer_;
    MappedFileManager file_manager_;
};
}  // namespace handy::camera
