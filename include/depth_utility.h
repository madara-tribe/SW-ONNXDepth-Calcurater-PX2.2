#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <optional>
class Inference {
public:
    Inference(const std::string& model_path, bool useCUDA);
    //cv::Mat run_inference(const cv::Mat& input_image);

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;

    int net_h = 384; // Replace with the actual height of the network input
    int net_w = 384; // Replace with the actual width of the network input
};

#endif // INFERENCE_H

