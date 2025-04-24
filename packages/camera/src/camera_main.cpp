#include "camera.h"

#include <condition_variable>
#include <csignal>
#include <iostream>
#include <mutex>

struct GlobalCameraRecorderInfo {
    std::mutex mutex;
    std::condition_variable condvar;
    handy::camera::CameraRecorder* camera_recorder_ptr = nullptr;
    bool ready_to_exit = false;
} global_recorder_info;

void handleSignal(int /*signum*/) {
    global_recorder_info.camera_recorder_ptr->stopInstance();
    sleep(3);
    global_recorder_info.ready_to_exit = true;
    global_recorder_info.condvar.notify_one();
}

// ./camera <param_launch_file> <output_mcap_filename>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("help\n./camera <param_launch_file> <output_mcap_filename>\n");
        return 0;
    }
    handy::camera::CameraRecorder writer(argv[1], argv[2], true);
    global_recorder_info.camera_recorder_ptr = &writer;
    // assign signal handler
    signal(SIGINT, handleSignal);  // Ctrl + C

    std::unique_lock<std::mutex> lock(global_recorder_info.mutex);
    while (!global_recorder_info.ready_to_exit) {  // check to handle spurious wakeups
        global_recorder_info.condvar.wait(lock);
    }

    printf("finished writing to file %s\n", argv[2]);

    return 0;
}
