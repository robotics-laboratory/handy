#include "CameraApi.h"

#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <thread>

#include <boost/lockfree/queue.hpp>

using namespace std::chrono_literals;

namespace handy::camera {

class CameraRecorder;

struct Size {
    size_t area() const { return static_cast<size_t>(width * height); };

    int width;
    int height;

    bool operator==(const Size& other) { return width == other.width && height == other.height; }
    bool operator!=(const Size& other) { return !(*this == other); }
};

struct BufferPair {
    uint8_t* first;
    uint8_t* second;
};

struct StampedImageBuffer {
    uint8_t* raw_buffer = nullptr;
    uint8_t* bgr_buffer = nullptr;
    tSdkFrameHead frame_info{};
    int camera_idx;
    int frame_id;
    uint32_t timestamp;
};

struct SynchronizedFrameBuffers {
    std::vector<BufferPair> images;
    std::vector<Size> image_sizes;
    uint32_t timestamp;  // in 0.1 milliseconds
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
    using CameraSubscriberCallback = std::function<void(std::shared_ptr<SynchronizedFrameBuffers>)>;

    CameraRecorder(const char* param_file, const char* output_filename, bool save_to_file = false);
    ~CameraRecorder();
    void registerSubscriberCallback(CameraSubscriberCallback callback);
    void stopInstance();

    constexpr static int kMaxCameraNum = 4;
    constexpr static int kQueueCapacity = 5;

  private:
    void handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info);
    void triggerCamera();
    void synchronizeQueues();
    static int getCameraId(int camera_handle);
    void applyParamsToCamera(int handle);
    void saveSynchronizedBuffers(std::shared_ptr<SynchronizedFrameBuffers> images);

    struct Params {
        std::chrono::duration<double> latency{50.0};  // in milliseconds
        std::string param_file;
        std::string output_filename;
        int fps = 20;
        int frames_to_take = 1000;
        int master_camera_id = 1;
        bool use_hardware_triger = false;
        bool save_to_file = false;
    } param_{};

    struct State {
        std::array<std::unique_ptr<boost::lockfree::queue<StampedImageBuffer>>, kMaxCameraNum>
            camera_images;
        std::array<std::unique_ptr<boost::lockfree::queue<BufferPair>>, kMaxCameraNum> free_buffers;

        std::atomic<bool> running = true;
        std::vector<std::thread> threads;  // trigger thread, queue handler thread
        std::vector<std::atomic<int>> frame_ids;
        std::vector<std::atomic<size_t>> current_buffer_idx;
        int camera_num = 2;
        std::vector<int> files;
        std::map<int, int> handle_to_idx;
        std::vector<Size> frame_sizes;
        std::vector<std::mutex> file_mutexes;
        std::vector<int> camera_handles;
        std::vector<void*> alligned_buffers;
        std::vector<CameraSubscriberCallback> registered_callbacks;
        CameraPool buffers;
        uint32_t initial_timestamp_difference = 0;
    } state_{};
};
}  // namespace handy::camera
