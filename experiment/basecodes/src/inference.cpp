#include <chrono>
//#include "utility.h"
#include "yolo_inference.h"
#include "midas_inference.h"

#define ONNX_DEPTH_PATH "/workspace/weights/dpt_large_384.onnx"
#define ONNX_YOLO_PATH "/workspace/weights/yolov7Tiny_640_640.onnx"
#define IMG_PATH "/workspace/data/indoor.jpg"
#define YOLO_INPUT_H 640
#define YOLO_INPUT_W 640


void yolo(){
    std::string model_path = ONNX_YOLO_PATH;
    YoloDetect yolo_detector(model_path);
    
    cv::Mat image = cv::imread(IMG_PATH);
    cv::Mat inputImage = yolo_detector.preprocess(image, YOLO_INPUT_H, YOLO_INPUT_W);

    
    std::vector<Ort::Value> outputTensors = yolo_detector.RunInference(inputImage);

    std::vector<Result> resultVector = yolo_detector.postprocess(image.size(), outputTensors);

    yolo_detector.drawBoundingBox(image, resultVector);
    cv::imwrite("resultt.png", image);
}

int main(int argc, char* argv[]) {
    bool useCUDA{false};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";

    if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0)) {
        useCUDA = true;
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    } else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0)) {
        useCUDA = false;
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    std::string onnx_model_path = ONNX_DEPTH_PATH;
    

    MidasInference midas(onnx_model_path, useCUDA);
    
    auto start = std::chrono::high_resolution_clock::now();
    midas.runInference(IMG_PATH);
    yolo();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Prediction took " << diff.count() << " seconds" << std::endl;
    return 0;
}
