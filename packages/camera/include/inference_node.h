#pragma once

#include "camera.h"
#include <torch/script.h>
#include <opencv2/core.hpp>
#include "c10/cuda/CUDAStream.h"
#include <boost/lockfree/queue.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <deque>
#include <vector>
#include <functional>
#include <memory>
#include <array>

namespace handy::camera {

class BallInferenceNode {
  public:
    static const int kMaxCameraNum = 2;

    /// Per‐frame detection result
    struct DetectionOutput {
        std::array<cv::Point, kMaxCameraNum> centers;  // one ball center per camera image
        std::array<bool, kMaxCameraNum> flags;         // true = needs segmentation for that camera
        uint32_t timestamp = 0;
        SynchronizedFrameBuffers* images_to_segment = nullptr;
    };

    /// Per‐frame segmentation result
    struct SegmentationOutput {
        // one centroid per camera image that requested segmentation
        std::vector<cv::Point> centroids;
        std::vector<int> camera_indices;
        uint32_t timestamp = 0;
    };

    using DetectionCallback = std::function<void(DetectionOutput)>;
    using SegmentationCallback = std::function<void(SegmentationOutput)>;

    /// ctor loads both models and sets up queues/threads
    BallInferenceNode(
        const std::string& det_model_path, const std::string& seg_model_path, Size input_size,
        std::function<void(SynchronizedFrameBuffers*)> deleter, int window_size = 5);
    ~BallInferenceNode();

    /// Subscribe to detection outputs
    /// Called from arbitrary thread; callbacks must be lock‐free/low‐latency.
    void registerDetectionSubscriber(const DetectionCallback& cb);

    /// Subscribe to segmentation outputs
    void registerSegmentationSubscriber(const SegmentationCallback& cb);

    /// Start the internal threads
    void start();

    /// Signal all threads to stop, join them, and release resources
    void stop();

    void handleSyncBuffer(std::shared_ptr<SynchronizedFrameBuffers> buffer);

  private:
    void detectSingleCamera(const std::deque<at::Tensor>& frames, DetectionOutput& detect_result,
        int camera_idx, c10::cuda::CUDAStream& infer_stream);
    void detectionLoop();
    void segmentationLoop();


    // === model objects ===
    torch::jit::script::Module det_model_;
    torch::jit::script::Module seg_model_;

    std::function<void(SynchronizedFrameBuffers*)> buffer_deleter_;

    struct Param {
        // === input queue & rolling window ===
        int window_size = 5;
        Size input_size{320, 192};
        const at::Tensor means =
            torch::tensor({0.077, 0.092, 0.142}).view({1, 3, 1, 1}).to(at::kFloat).to(at::kCUDA);
        const at::Tensor stds =
            torch::tensor({0.068, 0.079, 0.108}).view({1, 3, 1, 1}).to(at::kFloat).to(at::kCUDA);
    } param_;

    struct State {
        boost::lockfree::queue<SynchronizedFrameBuffers*> det_queue{64};

        // protected by ring_mutex_
        std::array<std::deque<at::Tensor>, kMaxCameraNum> ring_buffer;
        std::vector<at::Tensor> pinned_pool;
        std::atomic<size_t> pool_idx = 0;
        std::mutex ring_mutex;

        // === intermediate queues ===
        boost::lockfree::queue<DetectionOutput> seg_queue{128};

        // === subscriber callbacks ===
        std::vector<DetectionCallback> det_cbs;
        std::vector<SegmentationCallback> seg_cbs;
        std::mutex cb_mutex;

        // === threading & synchroization ===
        std::atomic<bool> running{false};
        std::thread det_thread;
        std::thread seg_thread;
        std::condition_variable seg_cv;
        std::mutex seg_mutex;
    } state_;
};

}  // namespace handy::camera
