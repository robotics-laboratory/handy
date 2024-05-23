#include "CameraApi.h"

#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>

using namespace std::chrono_literals;

namespace handy::camera {
class Writer {
  public:
    Writer(char* param_file, char* output_filename);

  private:
    void handleFrame(CameraHandle handle, BYTE* raw_buffer, tSdkFrameHead* frame_info);
    int getCameraId(int camera_handle);
    void applyParamsToCamera(int handle);

    struct Size {
        size_t area() const { return static_cast<size_t>(width * height); };

        int width;
        int height;
    };

    struct Params {
        std::chrono::duration<double> latency{50.0};  // in milliseconds
        std::string param_file;
        std::string output_filename;
        int fps = 20;
        int frames_to_take = 1000;
        int master_camera_id = 1;
        bool use_hardware_triger = false;
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
}  // namespace handy::camera
