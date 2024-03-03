#include <stdexcept>
#include <string>
#include <string_view>

#include "CameraApi.h"
#include "camera_status.h"

void abortIfNot(std::string_view msg, int status) {
    if (status != CAMERA_STATUS_SUCCESS) {
        const auto status_name = toStatusName(status);
        printf(
            "%.*s, %.*s(%i)", len(msg), msg.data(), len(status_name), status_name.data(), status);
        abort();
    }
}

void printHelp() {
    printf("\nhelp:\n");

    printf("usage: ./camera_identifier <camera ID>\n");
    printf("<camera ID> must be 8-bit unsigned int\n");
    printf("note that only one camera should be attached\n");
}

int main(int argc, char* argv[]) {
    uint8_t required_camera_id;
    if (argc == 2) {
        try {
            required_camera_id = static_cast<uint8_t>(std::stoi(argv[1]));
        } catch (const std::exception& e) {
            printf("%s -- failed to read required id: %s\n", e.what(), argv[1]);
            printHelp();
            return -1;
        }
    } else {
        printHelp();
        return -1;
    }

    printf("required id is: %d\n", required_camera_id);

    abortIfNot("camera init", CameraSdkInit(0));
    int camera_num = 100;  // attach all connected cameras
    tSdkCameraDevInfo cameras_list[100];
    abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &camera_num));
    if (camera_num != 1) {
        printf("more than 2 cameras are attached, assignment is ambiguous\n");
        printHelp();
        return -1;
    }

    int camera_handle;
    abortIfNot("camera init", CameraInit(cameras_list, -1, -1, &camera_handle));

    uint8_t prev_id;
    abortIfNot("get prev camera_id", CameraLoadUserData(camera_handle, 0, &prev_id, 1));
    printf("prev id=%d\n", prev_id);
    abortIfNot("set new camera_id", CameraSaveUserData(camera_handle, 0, &required_camera_id, 1));

    abortIfNot("camera uninit", CameraUnInit(camera_handle));
    printf("success\n");

    return 0;
}
