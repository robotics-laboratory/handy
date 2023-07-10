#include "CameraApi.h" //API header file of Camera SDK

#include "opencv2/core/core.hpp"
#include <opencv2/core/mat.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <fstream>
#include <iostream>

using namespace cv;

unsigned char *g_pRgbBuffer;     //Processed data cache

int main() {

    int iCameraCounts = 1;
    int iStatus = -1;
    tSdkCameraDevInfo tCameraEnumList;
    int hCamera;
    tSdkCameraCapbility tCapability;      //Device description information
    tSdkFrameHead sFrameInfo;
    BYTE *pbyBuffer;
    int iDisplayFrames = 10000;
    cv::Mat *iplImage = NULL;
    int channel = 3;

    CameraSdkInit(0); // 0 - is English language select

    //Enumerate devices and create a device list
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    printf("state = %d\n", iStatus);

    printf("count = %d\n", iCameraCounts);
    //no device connected
    if (iCameraCounts == 0) {
        return -1;
    }

    //The camera is initialized. After the initialization is successful,
    // any other camera-related operation interface can be called
    iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

    //initialization failed
    printf("state = %d\n", iStatus);
    if (iStatus != CAMERA_STATUS_SUCCESS) {
        return -1;
    }

    //Get the camera's characterization structure.
    // This structure contains the range information of various parameters that can be set by the camera.
    // Determines the parameters of the relevant function
    CameraGetCapability(hCamera, &tCapability);


    // возможные разрешение: 1280 * 1024 и 640 * 480 ROI
    // Camera output image format: Bayer GB 8bit (1Bpp) 17301514, последнее - код; Bayer GR 12bit Packed (1.5Bpp) 17563690

    //
    g_pRgbBuffer = (unsigned char *) malloc(
            tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);
    pbyBuffer = (unsigned char *) malloc(
            tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3 * 4);

    //g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    /*Put the SDK into working mode and start receiving images from the camera
     data. If the current camera is in trigger mode, it needs to receive
     The image is not updated until the frame is triggered.*/
    CameraPlay(hCamera);

    /*Other camera parameter settings
    For example CameraSetExposureTime   CameraGetExposureTime  Set/read exposure time
         CameraSetImageResolution  CameraGetImageResolution set/read resolution
         CameraSetGamma、CameraSetConrast、CameraSetGain Set the image gamma, contrast, etc.RGB  Digital gain and more.
         This routine is just to demonstrate how to convert the image obtained in the SDK,
         Convert to the image format of OpenCV, so as to call the image processing function of OpenCV for subsequent development
    */

    if (tCapability.sIspCapacity.bMonoSensor) {
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }


    //Cycle through 1000 frames of images
    while (iDisplayFrames--) {
        if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
            std::ofstream file("raw_data.bin", std::ios::binary);
            //file.write(data, 100);
            for (int i = 0 ; i < 1024*1280; ++i) {
                //std::cout << static_cast<int>(*(pbyBuffer + i)) << ' ';
                file << *(pbyBuffer + i);
            }

            std::cout << '\n';

            CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);

            cv::Mat matImage(
                    std::vector < int > {sFrameInfo.iHeight, sFrameInfo.iWidth},
                    sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3, // 8-bit unsigned 3-channel
                    g_pRgbBuffer
            );
            printf("%d  %d  \n", matImage.rows, matImage.cols);
            imwrite("test_1.png", matImage);

            break;

            waitKey(5);

            // After successfully calling CameraGetImageBuffer, you must call CameraReleaseImageBuffer to release the obtained buffer.
            //Otherwise, when calling CameraGetImageBuffer again, the program will be suspended and blocked until other threads call CameraReleaseImageBuffer to release the buffer
            CameraReleaseImageBuffer(hCamera, pbyBuffer);

        }
    }

    CameraUnInit(hCamera);
    //Note that after deinitialization, free
    free(g_pRgbBuffer);


    return 0;
}

// сначала читаем буфер и только затем преобразуем. То есть можем сразу передавать в топик сырой буфер, это будет быстрее
// одновременно можем преобразовывать и в стандартное rgb. По нему уже детекция и калибровка