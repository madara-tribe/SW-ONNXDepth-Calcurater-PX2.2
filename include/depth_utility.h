#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

class Inference {
public:
    Inference(const std::string& model_path, bool useCUDA);
    cv::Mat run_inference(const cv::Mat& input_image);

private:
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name;
    const char* output_name;
    int net_h = 384;
    int net_w = 384;
};

#endif // INFERENCE_H

