#ifndef MIDAS_INFERENCE_H
#define MIDAS_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class MidasInference {
public:
    MidasInference(const std::string& modelPath, bool useCUDA = false);
    ~MidasInference();

    cv::Mat PreProcess(cv::Mat& iImg);
    cv::Mat verifyOutput(float* output);
    void draw_depth(const cv::Mat& depth_map, int w, int h);
    void runInference(const char* imgPath);

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session* session;
    Ort::RunOptions run_options;
    int numthreads = 0;
    
    int H = 384;
    int W = 384;
    const char* input_node_name = "x.1";
    const char* output_node_name = "3195";
    
    const double set_min_depth = 0.0;
    const double set_max_depth = 100.0;
};

#endif // MIDAS_INFERENCE_H

