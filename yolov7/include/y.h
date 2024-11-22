#ifndef YOLO_INFERENCE_H
#define YOLO_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

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

std::vector<std::string> classNames = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

class YoloDetect {
public:
    YoloDetect(const std::string& modelPath);
    ~YoloDetect();
    float* RunSession(cv::Mat inputImage);
    cv::Mat preprocess(cv::Mat& image);
    std::vector<Result> postprocess(cv::Size originalImageSize, float* rawOutput);
    void drawBoundingBox(cv::Mat& image, std::vector<Result>& resultVector);

private:
    int model_input_width;
    int model_input_height;
    int pad_size_y;
    int pad_size_x;
    int model_width_after_padding;
    int model_height_after_padding;
    Ort::SessionOptions session_options;
    Ort::Env env;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Session* session;
};

#endif


