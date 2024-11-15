#include <iostream>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <fstream>
#include <vector>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define ONNX_DEPTH_PATH "/workspace/weights/dpt_large_384.onnx"
#define ONNX_YOLO_PATH "/workspace/weights/yolov7Tiny_640_640.onnx"
#define IMG_PATH "/workspace/input/input.jpg"


int model_check() {
    // Initialize the ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Model");
    std::string modelFilepath{ONNX_DEPTH_PATH};
    std::ifstream modelFile(modelFilepath);
    if (!modelFile.good()) {
        std::cerr << "Model file does not exist or could not be found: " << modelFilepath << std::endl;
    return -1;
    }
    // Create an ONNX Runtime session options object
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Load the model
    std::cout << "model name " << modelFilepath.c_str() << std::endl;
    Ort::Session session(env, modelFilepath.c_str(), session_options);
    // Get input and output details
    size_t num_input_nodes = session.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;

    for (size_t i = 0; i < num_input_nodes; i++) {
        // Updated to use GetInputNameAllocated
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << ": " << input_name.get() << std::endl;
    }
    std::cout << "Model loaded successfully!" << std::endl;
    return 0;
}



int main(int argc, char* argv[])
{
    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }
   
    return model_check(); //(useCUDA);
}
