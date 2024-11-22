#include <iostream>


#define ONNX_YOLO_PATH "/workspace/yolov7/model/yolov7-tiny.onnx"
#define IMG_PATH "/workspace/yolov7/data/dog.jpg"

#include "utility.h"
#include "yolo_inference.h"

int main()
{
    const char* model_path = ONNX_YOLO_PATH;
    
    YoloDetect yolo_detect(model_path);


    const char* input_node_name = "images";
    const char* output_node_name = "output";


    cv::Mat image = cv::imread(IMG_PATH);

    cv::Mat inputImage = yolo_detect.preprocess(image);
    float* Output = yolo_detect.RunSession(inputImage);
    std::vector<Result> resultVector = yolo_detect.postprocess(image.size(), Output);

    yolo_detect.drawBoundingBox(image, resultVector);
    cv::imwrite("results.png", image);

    return 0;
}

