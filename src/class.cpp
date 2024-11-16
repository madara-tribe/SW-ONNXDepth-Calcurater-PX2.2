#include "depth_utility.h"

Inference::Inference(const std::string& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    allocator = Ort::AllocatorWithDefaultOptions();
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    input_name = session->GetInputNameAllocated(0, allocator).get();
    output_name = session->GetOutputNameAllocated(0, allocator).get();

    auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    net_h = input_shape[2];
    net_w = input_shape[3];
}

cv::Mat Inference::run_inference(const cv::Mat& input_image) {
    // Preprocess image
    cv::Mat img_resized;
    cv::resize(input_image, img_resized, cv::Size(net_w, net_h));
    img_resized.convertTo(img_resized, CV_32F, 1.0 / 255);
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

    // Create input tensor
    std::vector<float> input_tensor_values((float*)img_resized.data, (float*)img_resized.data + img_resized.total() * img_resized.channels());
    std::array<int64_t, 4> input_shape{1, 3, net_h, net_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    auto output_data = output_tensors.front().GetTensorMutableData<float>();

    // Create output depth map
    cv::Mat depth_map(net_h, net_w, CV_32F, output_data);
    return depth_map.clone();  // Clone to prevent shared memory issues
}

