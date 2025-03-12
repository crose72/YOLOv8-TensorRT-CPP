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

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath, trtModelPath, inputVideo;
    if (!parseArgumentsVideo(argc, argv, config, onnxModelPath, trtModelPath, inputVideo)) {
        return -1;
    }

    YoloV8 yoloV8(onnxModelPath, trtModelPath, config);
    VPIBackend backend = VPI_BACKEND_CUDA;

    std::string gst_pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        "width=1280, height=720, framerate=30/1 ! nvvidconv flip-method=2 ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false";
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open CSI camera" << std::endl;
        return -1;
    }

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

    VPIArray inputBoxList = NULL, inputPredList = NULL;
    bool trackingActive = false;

    // Initialize KLT tracker
    VPIKLTFeatureTrackerParams kltFeatParams = {};

    // Set required parameters based on the available struct fields
    kltFeatParams.numberOfIterationsScaling = 10;   // Number of iterations for scale estimation
    kltFeatParams.nccThresholdUpdate = 0.85f;       // Threshold for requiring template update
    kltFeatParams.nccThresholdKill = 0.5f;          // Threshold to consider tracking lost
    kltFeatParams.nccThresholdStop = 0.95f;         // Early stopping threshold
    kltFeatParams.maxScaleChange = 0.2f;            // Max allowed scale change
    kltFeatParams.maxTranslationChange = 10.0f;     // Max allowed translation change
    kltFeatParams.trackingType = VPI_KLT_INVERSE_COMPOSITIONAL;  // Affine tracking (can use HOMOGRAPHY if needed)

    VPIKLTFeatureTrackerCreationParams kltParams;
    kltParams.maxTemplateCount = 28;
    kltParams.maxTemplateWidth = 128;
    kltParams.maxTemplateHeight = 128;
    VPIPayload kltTracker = NULL;
    CHECK_STATUS(vpiCreateKLTFeatureTracker(backend, 1280, 720, VPI_IMAGE_FORMAT_U8, &kltParams, &kltTracker));

    VPIArray outputBoxList = NULL, outputEstimationList = NULL;

    // Create output arrays for tracking results
    CHECK_STATUS(vpiArrayCreate(28, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &outputBoxList));  // 28 max tracked objects
    CHECK_STATUS(vpiArrayCreate(28, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &outputEstimationList));  // 28 transformations
    CHECK_STATUS(vpiArrayCreate(28, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &inputPredList));

    //VPIArrayBuffer buffer = NULL;
    VPIArrayBufferType bufferType = VPI_ARRAY_BUFFER_CUDA_AOS;

    while (true) {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        std::vector<VPIKLTTrackedBoundingBox> bboxes;
        std::vector<VPIHomographyTransform2D> preds;
        if (!trackingActive) {
            const std::vector<Object> objects = yoloV8.detectObjects(frame);
            std::cout << "Number of objects detected: " << objects.size() << std::endl;

            // Draw the bounding boxes on the image
            yoloV8.drawObjectLabels(frame, objects);
            
            // wrap opencv image in vpi image AFTER the yolo detector or things will break,
            // yolo is expecting a color image with 3 channels and grayscale only has 1.
            cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgReference, frame));
            for (const auto& obj : objects) {
                if (obj.label == 0) {
                    VPIKLTTrackedBoundingBox box = {};
                    box.bbox.xform.mat3[0][0] = 1;
                    box.bbox.xform.mat3[1][1] = 1;
                    box.bbox.xform.mat3[0][2] = obj.rect.x;
                    box.bbox.xform.mat3[1][2] = obj.rect.y;
                    box.bbox.width  = obj.rect.width;
                    box.bbox.height = obj.rect.height;
                    box.trackingStatus = 0;
                    box.templateStatus = 1;
                    bboxes.push_back(box);

                    VPIHomographyTransform2D xform = {};
                    xform.mat3[0][0] = 1;
                    xform.mat3[1][1] = 1;
                    xform.mat3[2][2] = 1;
                    preds.push_back(xform);

                    trackingActive = true;
                }
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

        if (trackingActive) {
            if (!imgTemplate) std::cerr << "ERROR: imgTemplate is NULL" << std::endl;
            if (!imgReference) std::cerr << "ERROR: imgReference is NULL" << std::endl;
            if (!inputBoxList) std::cerr << "ERROR: inputBoxList is NULL" << std::endl;
            if (!inputPredList) std::cerr << "ERROR: inputPredList is NULL" << std::endl;
            if (!outputBoxList) std::cerr << "ERROR: outputBoxList is NULL" << std::endl;
            if (!outputEstimationList) std::cerr << "ERROR: outputEstimationList is NULL" << std::endl;
            if (!kltTracker) std::cerr << "ERROR: kltTracker is NULL" << std::endl;

            CHECK_STATUS(vpiSubmitKLTFeatureTracker(stream, backend, kltTracker, imgTemplate, 
                                            inputBoxList, inputPredList, imgReference,
                                            outputBoxList, outputEstimationList, &kltFeatParams)); 


            CHECK_STATUS(vpiStreamSync(stream));  // Wait for tracking to finish

            // Get updated bounding boxes
            VPIArrayData data;
            CHECK_STATUS(vpiArrayLockData(inputBoxList, VPI_LOCK_READ, bufferType, &data));

            VPIKLTTrackedBoundingBox* trackedBoxes = (VPIKLTTrackedBoundingBox*)data.buffer.aos.data;
            int numTracked = *data.buffer.aos.sizePointer;

            std::cout << "KLT Tracker: " << numTracked << " objects tracked." << std::endl;

            // Update bounding boxes and check if tracking failed
            bboxes.clear();
            for (int i = 0; i < numTracked; ++i) {
                if (trackedBoxes[i].trackingStatus == 0) {  // 0 means successfully tracked
                    bboxes.push_back(trackedBoxes[i]);
                }
            }

            CHECK_STATUS(vpiArrayUnlock(inputBoxList));

            // If no objects were successfully tracked, re-enable YOLO detection
            if (bboxes.empty()) {
                std::cout << "Tracking lost. Re-enabling YOLO detection." << std::endl;
                trackingActive = false;
            }
        }


        for (const auto& box : bboxes) {
            cv::rectangle(frame, cv::Rect(box.bbox.xform.mat3[0][2], box.bbox.xform.mat3[1][2], box.bbox.width, box.bbox.height),
                          cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) >= 0) break;
    }

    vpiStreamDestroy(stream);
    vpiArrayDestroy(inputBoxList);
    vpiArrayDestroy(inputPredList);
    vpiImageDestroy(imgReference);
    vpiImageDestroy(imgTemplate);
    vpiArrayDestroy(outputBoxList);
    vpiArrayDestroy(outputEstimationList);
    vpiPayloadDestroy(kltTracker);


    return 0;
}
