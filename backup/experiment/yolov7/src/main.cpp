#define ONNX_YOLO_PATH "/workspace/yolov7/model/yolov7-tiny.onnx"
#define IMG_PATH "/workspace/yolov7/data/dog.jpg"
#define YOLO_INPUT_H 640
#define YOLO_INPUT_W 640
#include "yolo_inference.h"

int main(){
    std::string model_path = ONNX_YOLO_PATH;
    YoloDetect yolo_detector(model_path);
    
    cv::Mat image = cv::imread(IMG_PATH);
    cv::Mat inputImage = yolo_detector.preprocess(image, YOLO_INPUT_H, YOLO_INPUT_W);

    
    std::vector<Ort::Value> outputTensors = yolo_detector.RunInference(inputImage);

    std::vector<Result> resultVector = yolo_detector.postprocess(image.size(), outputTensors);

    yolo_detector.drawBoundingBox(image, resultVector);
    cv::imwrite("resultt.png", image);

    return 0;
}
