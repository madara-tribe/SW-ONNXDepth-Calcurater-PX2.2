#include <iostream>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <fstream>
#include <vector>
#define ONNX_MODEL_PATH "dpt_large_384.onnx"
#define IMG_PATH "input/input.jpg"

int onnx_predict(bool useCUDA){
    // Define data and model
    std::string instanceName{"cycleGAN inference"};
    std::string modelFilepath{ONNX_MODEL_PATH};
    // std::string imageFilepath{IMG_PATH};
    
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
    OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    
    return 0;
    }

int main(int argc, char* argv[])
{
    bool useCUDA{false};
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

    return onnx_predict(useCUDA);
}
