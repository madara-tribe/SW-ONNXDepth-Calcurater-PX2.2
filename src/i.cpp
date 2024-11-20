#include <iostream>
#include <cstring>
#include <chrono>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#define ONNX_DEPTH_PATH "/workspace/weights/dpt_large_384.onnx"
#define ONNX_YOLO_PATH "/workspace/weights/yolov7Tiny_640_640.onnx"
#define IMG_PATH "/workspace/data/indoor.jpg"
#define W 384
#define H 384
//#include "utility.h"
//#include "depth_utility.h"


void draw_depth(const cv::Mat& depth_map, int w, int h) {
    // Set minimum and maximum depth
    const double set_min_depth = 0.0;
    const double set_max_depth = 100.0;

    // Find minimum and maximum values in the depth map
    double min_depth, max_depth;
    cv::minMaxLoc(depth_map, &min_depth, &max_depth);

    // Update min_depth and max_depth based on set_min_depth and set_max_depth
    min_depth = std::min(min_depth, set_min_depth);
    max_depth = std::max(max_depth, set_max_depth);

    std::cout << "Min Depth: " << min_depth << ", Max Depth: " << max_depth << std::endl;

    // Normalize the depth map to the range [0, 255]
    cv::Mat norm_depth_map;
    norm_depth_map = 255.0 * (depth_map - min_depth) / (max_depth - min_depth);
    norm_depth_map = 255.0 - norm_depth_map;  // Invert the depth map

    // Convert to 8-bit image for applyColorMap
    cv::Mat abs_depth_map;
    norm_depth_map.convertTo(abs_depth_map, CV_8U);

    // Apply a color map to the normalized depth map
    cv::Mat color_depth;
    cv::applyColorMap(abs_depth_map, color_depth, cv::COLORMAP_JET);

    // Resize the depth map to match the input image shape
    cv::Mat resized_color_depth;
    cv::resize(color_depth, resized_color_depth, cv::Size(w, h));
    cv::imwrite("color_map2.png", resized_color_depth);
}

cv::Mat verifyOutput(float* output)
{
    cv::Mat segMat = cv::Mat::zeros(cv::Size(H, W), CV_8U);
    cv::Mat color_map = cv::Mat::zeros(cv::Size(H, W), CV_8U);
    // Populate image with output data
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            // Normalize the output to an appropriate value (assuming values between 0 and 255)
            segMat.at<uint8_t>(row, col) = static_cast<uint8_t>(*(output + (row * H) + col));
        }
    }
    cv::applyColorMap(segMat, color_map, cv::COLORMAP_JET);
    cv::imwrite("depth_map.png", segMat);
    cv::imwrite("color_map.png", color_map);
	return segMat;
}

bool BlobFromImage(cv::Mat& iImg, float* iBlob) {
    cv::Mat CHWImage;
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;
    
    cv::Mat channel_[3];
    cv::split(iImg, channel_);
    channel_[0] = (channel_[0] - 0.5) /0.5;
    channel_[1] = (channel_[1] - 0.5) /0.5;
    channel_[2] = (channel_[2] - 0.5) /0.5;
    cv::merge(channel_, 3, iImg);
    //cv::dnn::blobFromImage(iImg, CHWImage);
    //std::vector<float> chw_image(imgHeight * imgWidth * channels);
    
    std::cout << "total2 " << iImg.total() << std::endl;
    for (int c = 0; c < channels; c++){
        for (int h = 0; h < imgHeight; h++){
            for (int w = 0; w < imgWidth; w++){
              iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = static_cast<float>(iImg.at<cv::Vec3b>(h, w)[c]/255.0);
            }
        }
    }
    return true;
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_path>" << std::endl;
        return 1;
    }

    std::string onnx_model_path = ONNX_DEPTH_PATH;
    // Read input image
    cv::Mat img = cv::imread(IMG_PATH);
    if (img.empty()) {
        std::cerr << "Failed to read input image." << std::endl;
        return 1;
    }
    int inputHeight = img.rows;
    int inputWidth = img.cols;
    
    cv::Mat processedImg;
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, processedImg, cv::Size(H, W), cv::InterpolationFlags::INTER_CUBIC);
    float* blob = new float[H * W *3];
    std::cerr << "total1 " << processedImg.total() << std::endl;
    BlobFromImage(processedImg, blob);
     
    // Ort session 
    std::string modelFilepath{ONNX_DEPTH_PATH};
    
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Midas");
    Ort::SessionOptions sessionOption;
    if (useCUDA){
        OrtCUDAProviderOptions cuda_options{};
        sessionOption.AppendExecutionProvider_CUDA(cuda_options);    
    }
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(0);
    
    Ort::Session* session = new Ort::Session(env, modelFilepath.c_str(), sessionOption);
    
    const char* input_node_name = "x.1";
    const char* output_node_name = "3195";
    
    std::vector<int64_t> inputNodeDims = { 1, 3, H, W};
    Ort::MemoryInfo memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory_info, blob, 3 * H * W,inputNodeDims.data(), inputNodeDims.size());
    
    std::vector<float> output_data(1 * H * W);
    const std::vector<int64_t> output_shapes{1, H, W};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_data.data(), output_data.size(), output_shapes.data(), output_shapes.size());

    Ort::RunOptions run_options;
    session->Run(run_options, &input_node_name, &inputTensor, 1U, &output_node_name, &output_tensor, 1U);
    
    auto start = std::chrono::high_resolution_clock::now();
    float* output = output_tensor.GetTensorMutableData<float>();
    cv::Mat depth_map = verifyOutput(output);
    draw_depth(depth_map, inputWidth, inputHeight);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Prediction took " << diff.count() << " seconds" << std::endl;
    return 0; //(useCUDA);
}
