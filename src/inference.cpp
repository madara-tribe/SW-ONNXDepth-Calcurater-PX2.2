#include <chrono>
//#include "utility.h"
#include "yolo_inference.h"
#include "midas_inference.h"

#define ONNX_DEPTH_PATH "/workspace/weights/dpt_large_384.onnx"
#define ONNX_YOLO_PATH "/workspace/weights/yolov7Tiny_640_640.onnx"
#define IMG_PATH "/workspace/data/indoor.jpg"
#define YOLO_INPUT_H 640
#define YOLO_INPUT_W 640


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
    std::string yolo_model_path = ONNX_YOLO_PATH;
    std::string midas_model_path = ONNX_DEPTH_PATH;
    cv::Mat image = cv::imread(IMG_PATH);
    std::cout << "Midas Prepare" << std::endl;
    MidasInference midas(midas_model_path, useCUDA); // Midas instance
    std::cout << "YOLO Prepare" << std::endl;
    YoloDetect yolo_detector(yolo_model_path); // YOLO instance
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat midas_image = cv::imread(IMG_PATH);
    cv::Mat depth_colormap = midas.runInference(midas_image);
    
    cv::Mat yolo_image = cv::imread(IMG_PATH);
    cv::Mat inputImage = yolo_detector.preprocess(yolo_image, YOLO_INPUT_H, YOLO_INPUT_W);
    std::vector<Ort::Value> outputTensors = yolo_detector.RunInference(inputImage);
    std::vector<Result> resultVector = yolo_detector.postprocess(yolo_image.size(), outputTensors);
    cv::Mat yolo_result = yolo_detector.drawBoundingBox(yolo_image, resultVector);
    cv::imwrite("yolo_result.png", yolo_result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Prediction took " << diff.count() << " seconds" << std::endl;
    return 0;
}
