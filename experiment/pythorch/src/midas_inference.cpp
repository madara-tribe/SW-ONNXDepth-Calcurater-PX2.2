#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "utility.h"
#include "midas_inference.h"


void MidasInference(const std::string& modelPath, bool useCUDA){
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(numthreads);

    if (useCUDA) {
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
}

cv::Mat PreProcess(cv::Mat& iImg) {
    cv::cvtColor(iImg, iImg, cv::COLOR_BGR2RGB);
    cv::Mat resizedImage;
    cv::resize(iImg, iImg, cv::Size(W, H), cv::InterpolationFlags::INTER_CUBIC);
    cv::Mat channel_[3];
    cv::split(iImg, channel_);
    channel_[0] = (channel_[0] - 0.5) / 0.5;
    channel_[1] = (channel_[1] - 0.5) / 0.5;
    channel_[2] = (channel_[2] - 0.5) / 0.5;
    cv::merge(channel_, 3, iImg);
    
    cv::Mat floatImage;
    iImg.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    
    cv::Mat blobImage = cv::dnn::blobFromImage(floatImage);
    return blobImage;
}

cv::Mat verifyOutput(float* output) {
    cv::Mat segMat = cv::Mat::zeros(cv::Size(H, W), CV_8U);
    cv::Mat color_map = cv::Mat::zeros(cv::Size(H, W), CV_8U);
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            segMat.at<uint8_t>(row, col) = static_cast<uint8_t>(*(output + (row * H) + col));
        }
    }
    cv::applyColorMap(segMat, color_map, cv::COLORMAP_JET);
    cv::imwrite("depth_map.png", segMat);
    cv::imwrite("color_map.png", color_map);
    return segMat;
}

void draw_depth(const cv::Mat& depth_map, int w, int h) {
    double min_depth, max_depth;
    cv::minMaxLoc(depth_map, &min_depth, &max_depth);
    min_depth = std::min(min_depth, set_min_depth);
    max_depth = std::max(max_depth, set_max_depth);

    cv::Mat norm_depth_map;
    norm_depth_map = 255.0 * (depth_map - min_depth) / (max_depth - min_depth);
    norm_depth_map = 255.0 - norm_depth_map;

    cv::Mat color_depth;
    norm_depth_map.convertTo(color_depth, CV_8U);

    cv::applyColorMap(color_depth, color_depth, cv::COLORMAP_JET);

    cv::resize(color_depth, color_depth, cv::Size(w, h));
    cv::imwrite("color_map2.png", color_depth);
}

void runInference(const char* imgPath) {
    cv::Mat img = cv::imread(imgPath);
    
    int inputHeight = img.rows;
    int inputWidth = img.cols;
    
    cv::Mat blob = PreProcess(img);

    std::vector<int64_t> inputNodeDims = {1, 3, H, W};
    Ort::MemoryInfo memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total() * sizeof(float), inputNodeDims.data(), inputNodeDims.size());

    std::vector<float> output_data(1 * H * W);
    const std::vector<int64_t> output_shapes{1, H, W};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_data.data(), output_data.size(), output_shapes.data(), output_shapes.size());

    session->Run(run_options, &input_node_name, &inputTensor, 1U, &output_node_name, &output_tensor, 1U);

    float* output = output_tensor.GetTensorMutableData<float>();
    cv::Mat depth_map = verifyOutput(output);
    draw_depth(depth_map, inputWidth, inputHeight);
}
