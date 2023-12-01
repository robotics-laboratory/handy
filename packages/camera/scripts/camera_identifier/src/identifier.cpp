#include <string>
#include <stdexcept>
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("At least one parameter (camera ID) is required.\n");
        return 0;
    }

    uint8_t required_camera_id;
    try {
        required_camera_id = static_cast<uint8_t>(std::stoi(argv[1]));
    } catch (const std::exception& e) {
        printf("%s\n", e.what());
        return 0;
    }
    printf("Required camera ID is: %d\n", required_camera_id);

    abortIfNot("camera init", CameraSdkInit(0));
    int camera_num = 100;  // attach all connected cameras
    tSdkCameraDevInfo cameras_list[100];
    abortIfNot("camera listing", CameraEnumerateDevice(cameras_list, &camera_num));
    if (camera_num != 1) {
        printf("More than 2 cameras are attached. ID assignment is ambiguous.\n");
        return 0;
    }

    int camera_handle;
    abortIfNot("camera init", CameraInit(cameras_list, -1, -1, &camera_handle));

    uint8_t prev_id;
    abortIfNot("obtaining previous ID", CameraLoadUserData(camera_handle, 0, &prev_id, 1));
    printf("Obtained previous camera ID (may be undefined): %d\n", prev_id);

    abortIfNot("setting new ID", CameraSaveUserData(camera_handle, 0, &required_camera_id, 1));

    abortIfNot("obtaining newly set ID", CameraLoadUserData(camera_handle, 0, &prev_id, 1));
    printf("Set new camera ID: %d\n", prev_id);

    abortIfNot("camera uninit", CameraUnInit(camera_handle));

    return 0;
}