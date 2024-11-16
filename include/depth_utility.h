#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

class Inference {
public:
    Inference(const std::string& model_path);
    cv::Mat run_inference(const cv::Mat& input_image);

private:
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info;
    const char* input_name;
    const char* output_name;
    int net_h;
    int net_w;
};

#endif // INFERENCE_H

