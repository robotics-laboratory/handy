#include "camera.h"

#include <iostream>
#include <signal.h>
#include <mutex>
#include <condition_variable>

struct GlobalCameraRecorderInfo {
    std::mutex mutex;
    std::condition_variable condvar;
    handy::camera::CameraRecorder* camera_recorder_ptr = nullptr;
    bool ready_to_exit = false;
} global_recorder_info;

void handleSignal(int signum) {
    global_recorder_info.camera_recorder_ptr->stopInstance();
    global_recorder_info.ready_to_exit = true;
    global_recorder_info.condvar.notify_one();
}

// ./camera_bin <param_file> <output_filename>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("help\n./camera_bin <param_file> <output_filename>\n");
        return 0;
    }
    handy::camera::CameraRecorder writer(argv[1], argv[2], true);
    global_recorder_info.camera_recorder_ptr = &writer;
    signal(SIGINT, handleSignal);

    std::unique_lock<std::mutex> lock(global_recorder_info.mutex);
    while (!global_recorder_info.ready_to_exit) {
        global_recorder_info.condvar.wait(lock);
    }

    printf("finished writing to file %s\n", argv[2]);

    return 0;
}
