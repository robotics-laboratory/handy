#include <string>
#include <stdexcept>
#include <string_view>

#include <CameraApi.h>

std::string_view toStatusName(int status) {
    switch (status) {
        case CAMERA_STATUS_SUCCESS:
            return "SUCCESS";
        case CAMERA_STATUS_FAILED:
            return "FAILED";
        case CAMERA_STATUS_INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case CAMERA_STATUS_UNKNOW:
            return "UNKNOW";
        case CAMERA_STATUS_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case CAMERA_STATUS_NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case CAMERA_STATUS_PARAMETER_INVALID:
            return "PARAMETER_INVALID";
        case CAMERA_STATUS_PARAMETER_OUT_OF_BOUND:
            return "PARAMETER_OUT_OF_BOUND";
        case CAMERA_STATUS_UNENABLED:
            return "UNENABLED";
        case CAMERA_STATUS_USER_CANCEL:
            return "USER_CANCEL";
        case CAMERA_STATUS_PATH_NOT_FOUND:
            return "PATH_NOT_FOUND";
        case CAMERA_STATUS_SIZE_DISMATCH:
            return "SIZE_DISMATCH";
        case CAMERA_STATUS_TIME_OUT:
            return "TIME_OUT";
        case CAMERA_STATUS_IO_ERROR:
            return "IO_ERROR";
        case CAMERA_STATUS_COMM_ERROR:
            return "COMM_ERROR";
        case CAMERA_STATUS_BUS_ERROR:
            return "BUS_ERROR";
        case CAMERA_STATUS_NO_DEVICE_FOUND:
            return "NO_DEVICE_FOUND";
        case CAMERA_STATUS_NO_LOGIC_DEVICE_FOUND:
            return "NO_LOGIC_DEVICE_FOUND";
        case CAMERA_STATUS_DEVICE_IS_OPENED:
            return "DEVICE_IS_OPENED";
        case CAMERA_STATUS_DEVICE_IS_CLOSED:
            return "DEVICE_IS_CLOSED";
        case CAMERA_STATUS_DEVICE_VEDIO_CLOSED:
            return "DEVICE_VEDIO_CLOSED";
        case CAMERA_STATUS_NO_MEMORY:
            return "NO_MEMORY";
        case CAMERA_STATUS_FILE_CREATE_FAILED:
            return "FILE_CREATE_FAILED";
        case CAMERA_STATUS_FILE_INVALID:
            return "FILE_INVALID";
        case CAMERA_STATUS_WRITE_PROTECTED:
            return "WRITE_PROTECTED";
        case CAMERA_STATUS_GRAB_FAILED:
            return "GRAB_FAILED";
        case CAMERA_STATUS_LOST_DATA:
            return "LOST_DATA";
        case CAMERA_STATUS_EOF_ERROR:
            return "EOF_ERROR";
        case CAMERA_STATUS_BUSY:
            return "BUSY";
        case CAMERA_STATUS_WAIT:
            return "WAIT";
        case CAMERA_STATUS_IN_PROCESS:
            return "IN_PROCESS";
        case CAMERA_STATUS_IIC_ERROR:
            return "IIC_ERROR";
        case CAMERA_STATUS_SPI_ERROR:
            return "SPI_ERROR";
        case CAMERA_STATUS_USB_CONTROL_ERROR:
            return "USB_CONTROL_ERROR";
        case CAMERA_STATUS_USB_BULK_ERROR:
            return "USB_BULK_ERROR";
        case CAMERA_STATUS_SOCKET_INIT_ERROR:
            return "SOCKET_INIT_ERROR";
        case CAMERA_STATUS_GIGE_FILTER_INIT_ERROR:
            return "GIGE_FILTER_INIT_ERROR";
        case CAMERA_STATUS_NET_SEND_ERROR:
            return "NET_SEND_ERROR";
        case CAMERA_STATUS_DEVICE_LOST:
            return "DEVICE_LOST";
        case CAMERA_STATUS_DATA_RECV_LESS:
            return "DATA_RECV_LESS";
        case CAMERA_STATUS_FUNCTION_LOAD_FAILED:
            return "FUNCTION_LOAD_FAILED";
        case CAMERA_STATUS_CRITICAL_FILE_LOST:
            return "CRITICAL_FILE_LOST";
        case CAMERA_STATUS_SENSOR_ID_DISMATCH:
            return "SENSOR_ID_DISMATCH";
        case CAMERA_STATUS_OUT_OF_RANGE:
            return "OUT_OF_RANGE";
        case CAMERA_STATUS_REGISTRY_ERROR:
            return "REGISTRY_ERROR";
        case CAMERA_STATUS_ACCESS_DENY:
            return "ACCESS_DENY";
        case CAMERA_STATUS_CAMERA_NEED_RESET:
            return "CAMERA_NEED_RESET";
        case CAMERA_STATUS_ISP_MOUDLE_NOT_INITIALIZED:
            return "ISP_MOUDLE_NOT_INITIALIZED";
        case CAMERA_STATUS_ISP_DATA_CRC_ERROR:
            return "ISP_DATA_CRC_ERROR";
        case CAMERA_STATUS_MV_TEST_FAILED:
            return "MV_TEST_FAILED";
        case CAMERA_STATUS_INTERNAL_ERR1:
            return "INTERNAL_ERR1";
        case CAMERA_STATUS_U3V_NO_CONTROL_EP:
            return "U3V_NO_CONTROL_EP";
        case CAMERA_STATUS_U3V_CONTROL_ERROR:
            return "U3V_CONTROL_ERROR";
        case CAMERA_STATUS_INVALID_FRIENDLY_NAME:
            return "INVALID_FRIENDLY_NAME";
        case CAMERA_STATUS_FORMAT_ERROR:
            return "FORMAT_ERROR";
        case CAMERA_STATUS_PCIE_OPEN_ERROR:
            return "PCIE_OPEN_ERROR";
        case CAMERA_STATUS_PCIE_COMM_ERROR:
            return "PCIE_COMM_ERROR";
        case CAMERA_STATUS_PCIE_DDR_ERROR:
            return "PCIE_DDR_ERROR";
        default:
            throw std::runtime_error("Unknwown status!");
    }
}

inline int len(const std::string_view& s) { return static_cast<int>(s.size()); }

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