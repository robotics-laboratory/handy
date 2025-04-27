#pragma once

#include "camera.h"      
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <boost/lockfree/queue.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <deque>
#include <vector>
#include <functional>
#include <memory>

namespace handy::camera {

class BallInferenceNode {
  public:
    /// Per‐frame detection result
    struct DetectionOutput {
        std::vector<cv::Rect> bboxes;  // one bbox per camera image
        std::vector<bool> flags;       // true = needs segmentation for that camera
        uint32_t timestamp;
    };

    /// Per‐frame segmentation result
    struct SegmentationOutput {
        // one mask per camera image that requested segmentation
        std::vector<at::Tensor> masks;
        std::vector<int> camera_indices;
        uint32_t timestamp;
    };

    using DetectionCallback = std::function<void(std::shared_ptr<DetectionOutput>)>;
    using SegmentationCallback = std::function<void(std::shared_ptr<SegmentationOutput>)>;

    /// ctor loads both models and sets up queues/threads
    BallInferenceNode(
        const std::string& det_model_path, const std::string& seg_model_path, Size input_size,
        int window_size = 5);
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

  private:
    void detectionLoop();
    void segmentationLoop();

    // === model objects ===
    torch::jit::script::Module det_model_;
    torch::jit::script::Module seg_model_;

    // === input queue & rolling window ===
    boost::lockfree::queue<std::shared_ptr<SynchronizedFrameBuffers>> det_queue_{128};
    const int window_size_;
    Size input_size_;

    // protected by ring_mutex_
    std::deque<at::Tensor> ring_buffer_;
    std::mutex ring_mutex_;

    // === intermediate queues ===
    boost::lockfree::queue<DetectionOutput> seg_queue_{128};

    // === subscriber callbacks ===
    std::vector<DetectionCallback> det_cbs_;
    std::vector<SegmentationCallback> seg_cbs_;
    std::mutex cb_mutex_;

    // === threading & synchronization ===
    std::thread det_thread_;
    std::thread seg_thread_;
    std::atomic<bool> running_{false};
    std::condition_variable seg_cv_;
    std::mutex seg_mutex_;
};

}  // namespace handy::camera
