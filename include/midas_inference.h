#ifndef MIDAS_INFERENCE_H
#define MIDAS_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class MidasInference {
public:
    MidasInference(const std::string& modelPath, bool useCUDA = false);
    ~MidasInference();

    bool BlobFromImage(cv::Mat& iImg, float* iBlob);
    cv::Mat verifyOutput(float* output);
    void draw_depth(const cv::Mat& depth_map, int w, int h);
    void runInference(const cv::Mat& inputImage);

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session* session;
    int H = 384;
    int W = 384;
    const char* input_node_name = "x.1";
    const char* output_node_name = "3195";
    
    bool useCUDA;
};

#endif // MIDAS_INFERENCE_H

