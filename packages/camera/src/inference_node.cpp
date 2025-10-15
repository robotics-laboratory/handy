#include "inference_node.h"
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/types.h>
#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAGuard.h"
#include <ATen/cuda/CUDAContext.h>

namespace handy::camera {

BallInferenceNode::BallInferenceNode(
    const std::string& det_model_path, const std::string& seg_model_path, Size input_size,
    std::function<void(SynchronizedFrameBuffers*)> deleter, int window_size) {
    det_model_ = torch::jit::load(det_model_path);
    det_model_.eval();
    det_model_.to(at::kCUDA);

    seg_model_ = torch::jit::load(seg_model_path);
    seg_model_.eval();
    seg_model_.to(at::kCUDA);

    // Store params
    param_.input_size = input_size;
    param_.window_size = window_size;
    buffer_deleter_ = deleter;

    for (int i = 0; i < param_.window_size * kMaxCameraNum; ++i) {
        // Allocate an empty pinned tensor of shape [1, C, H, W]
        at::Tensor t = torch::empty(
                           {1, 3, param_.input_size.height, param_.input_size.width},
                           at::TensorOptions().dtype(torch::kUInt8))
                           .pin_memory();
        state_.pinned_pool.push_back(std::move(t));
    }
}

BallInferenceNode::~BallInferenceNode() { stop(); }

void BallInferenceNode::registerDetectionSubscriber(const DetectionCallback& cb) {
    std::lock_guard<std::mutex> lk(state_.cb_mutex);
    state_.det_cbs.push_back(cb);
}

void BallInferenceNode::registerSegmentationSubscriber(const SegmentationCallback& cb) {
    std::lock_guard<std::mutex> lk(state_.cb_mutex);
    state_.seg_cbs.push_back(cb);
}

void BallInferenceNode::handleSyncBuffer(std::shared_ptr<SynchronizedFrameBuffers> buffer) {
    while (!state_.det_queue.push(buffer.get())) {
    }
}

void BallInferenceNode::start() {
    state_.running = true;
    // Launch threads
    state_.det_thread = std::thread(&BallInferenceNode::detectionLoop, this);
    state_.seg_thread = std::thread(&BallInferenceNode::segmentationLoop, this);
}

void BallInferenceNode::stop() {
    bool expected = true;
    if (state_.running.compare_exchange_strong(expected, false)) {
        state_.seg_cv.notify_one();
        if (state_.det_thread.joinable()) {
            state_.det_thread.join();
        }
        if (state_.seg_thread.joinable()) {
            state_.seg_thread.join();
        }
    }
}

void BallInferenceNode::detectSingleCamera(
    const std::deque<at::Tensor>& frames, DetectionOutput& detect_result, int camera_idx,
    c10::cuda::CUDAStream& infer_stream) {
    std::vector<at::Tensor> frames_vec(frames.begin(), frames.end());
    at::Tensor batch = at::cat(frames_vec, 1);  // dim=1

    torch::NoGradGuard no_grad;
    at::cuda::CUDAStreamGuard guard(infer_stream);
    auto out_tuple = det_model_.forward({batch}).toTuple();

    // Unpack model outputs
    //    out_tuple[0]: prob distribution (1, width + height)
    //    out_tuple[1]: presence logits (1, 2)
    at::Tensor center_distr = out_tuple->elements()[0].toTensor();
    at::Tensor success_detection_distr = out_tuple->elements()[1].toTensor();

    // Threshold & argmax to get (x,y) center
    //    replicate: pred[pred < thresh]=0, then argmax on slices
    const float thresh = 0.0001;
    int W = param_.input_size.width;
    center_distr = center_distr.clone();
    center_distr.masked_fill_(center_distr < thresh, 0.0f);

    at::Tensor prob_x = center_distr.slice(1, 0, W);                     // {1, W}
    at::Tensor prob_y = center_distr.slice(1, W, center_distr.size(1));  // {1, H}

    int64_t pred_x = prob_x.argmax(1).item<int64_t>();
    int64_t pred_y = prob_y.argmax(1).item<int64_t>();
    detect_result.centers[camera_idx] = {pred_x, pred_y};

    int64_t pres_idx = success_detection_distr.argmax(1).item<int64_t>();
    detect_result.flags[camera_idx] = (pres_idx == 1);  // class 1 means object present
}

void BallInferenceNode::detectionLoop() {
    auto infer_stream = at::cuda::getStreamFromPool();
    std::vector<std::deque<at::Tensor>> rollBuf(kMaxCameraNum);

    SynchronizedFrameBuffers* sync;
    while (state_.running) {
        if (!state_.det_queue.pop(sync)) {
            std::this_thread::sleep_for(1ms);
            continue;
        }
        std::cout << "recieved buffer for detection\n";

        DetectionOutput detection_out;
        bool any_successful_detection = false;
        auto start = std::chrono::steady_clock::now();

        // Preprocess each camera image
        for (size_t cam = 0; cam < sync->images.size(); ++cam) {
            uint8_t* raw = sync->images[cam];
            Size sz = sync->image_sizes[cam];
            cv::Mat bayer(sz.height, sz.width, CV_8UC1, raw);

            

            static thread_local cv::Mat rgb;
            cv::cvtColor(bayer, rgb, cv::COLOR_BayerBG2RGB);

            static thread_local cv::Mat resized;
            cv::resize(rgb, resized, cv::Size(param_.input_size.width, param_.input_size.height));
            // To tensor  of shape (1, height, width, channel) and perform all the preprocessing
            at::Tensor t_uint8 = state_.pinned_pool[state_.pool_idx++ % state_.pinned_pool.size()];
            std::memcpy(
                t_uint8.data_ptr<uint8_t>(),  // pinned host memory
                resized.data,                 // OpenCV RGB data
                3 * param_.input_size.area());
            at::Tensor t = t_uint8.to(at::kCUDA, true)
                               .toType(at::kFloat)
                               .div_(255.0)
                               .sub_(param_.means)
                               .div_(param_.stds);

            rollBuf[cam].push_back(t);
            if (rollBuf[cam].size() > (size_t)param_.window_size) {
                rollBuf[cam].pop_front();
            }
            if (rollBuf[cam].size() < (size_t)param_.window_size) {
                continue;
            }
            detectSingleCamera(rollBuf[cam], detection_out, cam, infer_stream);
            any_successful_detection |= detection_out.flags[cam];
        }
        auto end = std::chrono::steady_clock::now();

        std::cout << "elapsed: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << '\n';
        detection_out.images_to_segment = nullptr;
        detection_out.timestamp = sync->timestamp;

        {
            std::lock_guard<std::mutex> lk(state_.cb_mutex);
            for (auto& cb : state_.det_cbs) {
                cb(detection_out);
            }
        }

        if (any_successful_detection) {
            // segmentation requires initial images, so we need to send pointer to buffer too
            detection_out.images_to_segment = sync;
            state_.seg_queue.push(detection_out);
            state_.seg_cv.notify_one();
        } else {
            // in case no segmentation is required we can release buffer
            buffer_deleter_(sync);
        }
    }
}

void BallInferenceNode::segmentationLoop() {
    while (state_.running) {
        std::unique_lock<std::mutex> lk(state_.seg_mutex);
        state_.seg_cv.wait_for(
            lk, 5ms, [this]() { return !state_.running || !state_.seg_queue.empty(); });
        if (!state_.running && state_.seg_queue.empty()) break;

        DetectionOutput det;
        if (!state_.seg_queue.pop(det)) {
            continue;
        }

        auto sync = det.images_to_segment;
        if (!sync) {
            continue;
        }

        std::vector<at::Tensor> masks;
        std::vector<int> cams;
        std::vector<cv::Point> centers;
        std::vector<cv::Point2f> centroids;

        SegmentationOutput segmentation_output;
        segmentation_output.timestamp = det.timestamp;

        // Process each flagged camera
        for (size_t cam = 0; cam < det.flags.size(); ++cam) {
            if (!det.flags[cam]) continue;

            uint8_t* raw = sync->images[cam];
            Size sz = sync->image_sizes[cam];
            cv::Mat bayer(sz.height, sz.width, CV_8UC1, raw);
            cv::Mat rgb;
            cv::cvtColor(bayer, rgb, cv::COLOR_BayerBG2RGB);

            const int cropSize = 128;
            int half = cropSize / 2;
            cv::Point pt = det.centers[cam];
            int x = std::clamp(pt.x, half, sz.width - half);
            int y = std::clamp(pt.y, half, sz.height - half);
            int x0 = x - half;
            int y0 = y - half;
            cv::Rect cropRect(x0, y0, cropSize, cropSize);
            cv::Mat crop = rgb(cropRect);

            at::Tensor t = torch::from_blob(crop.data, {1, crop.rows, crop.cols, 3}, at::kByte)
                               .permute({0, 3, 1, 2})
                               .toType(at::kFloat)
                               .div_(255.0)
                               .sub_(param_.means)
                               .div_(param_.stds)
                               .pin_memory();

            at::Tensor gpu_in = t.to(at::kCUDA, /*non_blocking=*/true);
            torch::NoGradGuard ndg;
            at::Tensor out = seg_model_.forward({gpu_in}).toTensor().cpu();
            // out shape {1, 1, cropSize, cropSize}
            at::Tensor prob = out.sigmoid().squeeze(0).squeeze(0);  // {H, W} on CUDA

            constexpr float thresh = 0.5f;
            at::Tensor bin = prob > thresh;  // ByteTensor {H, W} on CUDA
            at::Tensor coords = bin.nonzero();

            // compute centroid as the mean of these coords
            at::Tensor centroid;
            if (coords.numel() == 0) {
                // {row_center, col_center}
                centroid =
                    torch::tensor({cropSize / 2.0, cropSize / 2.0}, at::kFloat).to(at::kCUDA);
            } else {
                // convert coords to float and take mean over dim=0 -> {mean_row, mean_col}
                centroid = coords.to(at::kFloat).mean(0);
            }

            centroid = centroid.to(at::kCPU);

            float centroid_y = centroid[0].item<float>() + y0;
            float centroid_x = centroid[1].item<float>() + x0;

            segmentation_output.centroids.emplace_back(centroid_x, centroid_y);
            segmentation_output.camera_indices.push_back(cam);
        }

        if (segmentation_output.camera_indices.empty()) {
            continue;
        }

        std::lock_guard<std::mutex> lk_callback(state_.seg_mutex);
        for (auto& cb : state_.seg_cbs) {
            cb(segmentation_output);
        }

        // now we do not need buffer
        buffer_deleter_(sync);
    }
}

}  // namespace handy::camera
