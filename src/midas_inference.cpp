#include "midas_inference.h"
#include <iostream>
#include <cstring>
#include <chrono>


MidasInference::MidasInference(const std::string& modelPath, bool useCUDA)
    : env(ORT_LOGGING_LEVEL_WARNING, "Midas"), useCUDA(useCUDA) {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(0);

    if (useCUDA) {
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
}

MidasInference::~MidasInference() {
    delete session;
}

bool MidasInference::BlobFromImage(cv::Mat& iImg, float* iBlob) {
    cv::Mat channel_[3];
    cv::split(iImg, channel_);
    channel_[0] = (channel_[0] - 0.5) / 0.5;
    channel_[1] = (channel_[1] - 0.5) / 0.5;
    channel_[2] = (channel_[2] - 0.5) / 0.5;
    cv::merge(channel_, 3, iImg);

    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWidth; w++) {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = static_cast<float>(iImg.at<cv::Vec3b>(h, w)[c] / 255.0);
            }
        }
    }
    return true;
}

cv::Mat MidasInference::verifyOutput(float* output) {
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

void MidasInference::draw_depth(const cv::Mat& depth_map, int w, int h) {
    const double set_min_depth = 0.0;
    const double set_max_depth = 100.0;

    double min_depth, max_depth;
    cv::minMaxLoc(depth_map, &min_depth, &max_depth);
    min_depth = std::min(min_depth, set_min_depth);
    max_depth = std::max(max_depth, set_max_depth);

    cv::Mat norm_depth_map;
    norm_depth_map = 255.0 * (depth_map - min_depth) / (max_depth - min_depth);
    norm_depth_map = 255.0 - norm_depth_map;

    cv::Mat abs_depth_map;
    norm_depth_map.convertTo(abs_depth_map, CV_8U);

    cv::Mat color_depth;
    cv::applyColorMap(abs_depth_map, color_depth, cv::COLORMAP_JET);

    cv::Mat resized_color_depth;
    cv::resize(color_depth, resized_color_depth, cv::Size(w, h));
    cv::imwrite("color_map2.png", resized_color_depth);
}

void MidasInference::runInference(const cv::Mat& inputImage) {
    int inputHeight = inputImage.rows;
    int inputWidth = inputImage.cols;

    cv::Mat processedImg;
    cv::cvtColor(inputImage, processedImg, cv::COLOR_BGR2RGB);
    cv::resize(processedImg, processedImg, cv::Size(W, H), cv::InterpolationFlags::INTER_CUBIC);

    float* blob = new float[H * W * 3];
    BlobFromImage(processedImg, blob);

    std::vector<int64_t> inputNodeDims = {1, 3, H, W};
    Ort::MemoryInfo memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory_info, blob, 3 * H * W, inputNodeDims.data(), inputNodeDims.size());

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

    delete[] blob;
}


