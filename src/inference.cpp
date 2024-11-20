#include <iostream>
#include <cstring>
#include <chrono>

#define ONNX_DEPTH_PATH "/workspace/weights/dpt_large_384.onnx"
#define ONNX_YOLO_PATH "/workspace/weights/yolov7Tiny_640_640.onnx"
#define IMG_PATH "/workspace/data/indoor.jpg"

#include "midas_inference.h"

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
    cv::Mat img = cv::imread(IMG_PATH);
    if (img.empty()) {
        std::cerr << "Failed to read input image." << std::endl;
        return 1;
    }

    MidasInference midas(onnx_model_path, useCUDA);
    
    auto start = std::chrono::high_resolution_clock::now();
    midas.runInference(img);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Prediction took " << diff.count() << " seconds" << std::endl;
    return 0;
}
