#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

extern std::vector<std::string> classNames; // Use extern to declare the variable


typedef struct Result {
    int x1;
    int x2;
    int y1;
    int y2;
    int obj_id;
    float accuracy;

    Result(int x1_, int x2_, int y1_, int y2_, int obj_id_, float accuracy_) {
       x1 = x1_;
       x2 = x2_;
       y1 = y1_;
       y2 = y2_;
       obj_id = obj_id_;
       accuracy = accuracy_;
   }

} result_t ;



class YoloDetect {
public:
    YoloDetect(const std::string& modelPath);
    
    void drawBoundingBox(cv::Mat& image, std::vector<Result>& resultVector);
    cv::Mat preprocess(cv::Mat& image, int model_input_width, int model_input_height);
    std::vector<Result> postprocess(cv::Size originalImageSize, std::vector<Ort::Value>& outputTensors);
    std::vector<Ort::Value> RunInference(cv::Mat& inputImage);
private:
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    Ort::Env env;
    Ort::Session* session;
    int numthreads = 0;
    const char* input_node_name = "images";
    const char* output_node_name = "output";
    
    int pad_size_y;
    int pad_size_x;
    int model_width_after_padding;
    int model_height_after_padding;
};
