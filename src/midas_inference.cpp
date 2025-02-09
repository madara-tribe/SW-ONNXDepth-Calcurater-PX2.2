#include "midas_inference.h"


MidasInference::MidasInference(const std::string& modelPath, bool useCUDA){
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(numthreads);

    if (useCUDA) {
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
}

MidasInference::~MidasInference() {
    delete session;
}

std::vector<float> MidasInference::PreProcess(cv::Mat& iImg) {
    cv::cvtColor(iImg, iImg, cv::COLOR_BGR2RGB);
    cv::resize(iImg, iImg, cv::Size(W, H), cv::InterpolationFlags::INTER_CUBIC);
    // to 1 dim
    iImg = iImg.reshape(1, 1);
    std::vector<float> vec;
    iImg.convertTo(vec, CV_32FC1, 1. / 255);
    // HWC -> CHW
    std::vector<float> output;
    for (size_t ch = 0; ch<3; ++ch){
        for (size_t i = ch; i < vec.size(); i +=3){
            output.emplace_back((vec[i]-0.5) / 0.5);
        }
    }
    return output;
}

cv::Mat MidasInference::verifyOutput(std::vector<float> output) {
    cv::Mat segMat = cv::Mat::zeros(cv::Size(H, W), CV_8U);
    cv::Mat color_map = cv::Mat::zeros(cv::Size(H, W), CV_8U);
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            segMat.at<uint8_t>(row, col) = static_cast<uint8_t>(output[row * W + col]);
        }
    }
    cv::applyColorMap(segMat, color_map, cv::COLORMAP_JET);
    cv::imwrite("depth_map.png", segMat);
    return segMat;
}

cv::Mat MidasInference::draw_depth(const cv::Mat& depth_map, int oriW, int oriH) {
    float min_val = 0;
    float max_val = 100;

    float range_val = max_val - min_val;
    std::cout << "min max" << max_val << min_val << std::endl;
    cv::Mat norm_depth_map = ((depth_map - min_val) / range_val) * 255.0;
    norm_depth_map = 255 - norm_depth_map;
    
    cv::Mat color_depth;
    cv::applyColorMap(norm_depth_map, color_depth, cv::COLORMAP_JET);
    cv::resize(color_depth, color_depth, cv::Size(oriW, oriH));
    cv::imwrite("color_map.jpg", color_depth);
    return color_depth;
}

cv::Mat MidasInference::runInference(cv::Mat& img) {
    int inputHeight = img.rows;
    int inputWidth = img.cols;
    int insize = H * W * 3;
    int outsize = H * W * 1;
    // vector for input and output
    std::vector<float> input(insize);
    std::vector<float> results(outsize);
    
    // preprocess input image
    std::vector<float> blob = PreProcess(img);
    assert(blob.size() == insize);
    // copy images to input vector
    std::copy(blob.begin(), blob.end(), input.begin());
    // create input Tenspr
    std::vector<int64_t> inputNodeDims = {1, 3, H, W};
    Ort::MemoryInfo memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputNodeDims.data(), inputNodeDims.size());
    // create output Tenspr
    const std::vector<int64_t> output_shapes{1, H, W};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), output_shapes.data(), output_shapes.size());
    // Inference
    session->Run(run_options, &input_node_name, &inputTensor, 1U, &output_node_name, &output_tensor, 1U);

    cv::Mat depth_map = verifyOutput(results);
    cv::Mat color_depth = draw_depth(depth_map, inputWidth, inputHeight);
    return color_depth;
}

