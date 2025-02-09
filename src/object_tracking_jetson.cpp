#include "cmd_line_util.h"
#include "yolov8.h"
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <opencv2/opencv.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/KLTFeatureTracker.h>

#include <iostream>
#include <vector>
#include <map>

#define CHECK_STATUS(STMT) \
    do { \
        VPIStatus status = (STMT); \
        if (status != VPI_SUCCESS) { \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
            vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
            throw std::runtime_error(vpiStatusGetName(status) + std::string(": ") + buffer); \
        } \
    } while (0)

// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string trtModelPath;
    std::string inputVideo;

    // Parse the command line arguments
    if (!parseArgumentsVideo(argc, argv, config, onnxModelPath, trtModelPath, inputVideo)) {
        return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, trtModelPath, config);

    VPIBackend backend;
    backend = VPI_BACKEND_CUDA;

    // Define GStreamer pipeline for the CSI camera
    std::string gst_pipeline = 
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        "width=1280, height=720, framerate=30/1 ! nvvidconv flip-method=2 ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false";

    // Open the CSI camera
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open CSI camera" << std::endl;
        return -1;
    }

    std::cout << "CSI Camera opened successfully!" << std::endl;

    VPIStream stream = NULL;
    CHECK_STATUS(vpiStreamCreate(backend, &stream));

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Empty frame captured." << std::endl;
        return 1;
    }

    cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    VPIImage imgTemplate = NULL, imgReference = NULL;
    CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(frame, 0, &imgTemplate));
    CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(frame, 0, &imgReference));

    std::vector<VPIKLTTrackedBoundingBox> bboxes;
    std::vector<VPIHomographyTransform2D> preds;
    VPIArray inputBoxList = NULL, inputPredList = NULL;

    while (true) {
        // Grab frame
        
        cap >> frame;

        if (frame.empty())
            throw std::runtime_error("Unable to decode image from video stream.");

        // Run inference
        const std::vector<Object> objects = yoloV8.detectObjects(frame);

        cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgReference, frame));

        bboxes.clear();
        preds.clear();
        for (int i = 0; i < objects.size(); ++i)
        {
            if (objects[i].label == 0)
            {
                VPIKLTTrackedBoundingBox box = {};
                box.bbox.xform.mat3[0][0] = 1;
                box.bbox.xform.mat3[1][1] = 1;
                box.bbox.xform.mat3[0][2] = objects[i].rect.x;
                box.bbox.xform.mat3[1][2] = objects[i].rect.y;
                box.bbox.width  = objects[i].rect.width;
                box.bbox.height = objects[i].rect.height;
                box.trackingStatus = 0;
                box.templateStatus = 1;
                bboxes.push_back(box);

                VPIHomographyTransform2D xform = {};
                xform.mat3[0][0] = 1;
                xform.mat3[1][1] = 1;
                xform.mat3[2][2] = 1;
                preds.push_back(xform);
            }
        }

        if (!bboxes.empty()) {
            VPIArrayData data = {};
            data.bufferType = VPI_ARRAY_BUFFER_HOST_AOS;
            data.buffer.aos.type = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
            data.buffer.aos.capacity = bboxes.capacity();
            data.buffer.aos.sizePointer = new int32_t(bboxes.size());
            data.buffer.aos.data = bboxes.data();
            CHECK_STATUS(vpiArrayCreateWrapper(&data, 0, &inputBoxList));
        }

        // Draw tracked bounding boxes
        for (const auto& box : bboxes) {
            cv::rectangle(frame, cv::Rect(box.bbox.xform.mat3[0][2], box.bbox.xform.mat3[1][2], box.bbox.width, box.bbox.height),
                          cv::Scalar(0, 255, 0), 2);
        }

        // Draw the bounding boxes on the image
        //yoloV8.drawObjectLabels(frame, objects);

        // Display the results
        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) >= 0)
            break;
    }

    vpiStreamDestroy(stream);
    vpiArrayDestroy(inputBoxList);
    vpiArrayDestroy(inputPredList);
    vpiImageDestroy(imgReference);
    vpiImageDestroy(imgTemplate);

    return 0;
}