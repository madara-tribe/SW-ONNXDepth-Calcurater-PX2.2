#include "depth_utility.h"

Inference::Inference(const std::string& model_path, bool useCUDA) {
    std::string modelFilepath{model_path};
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    // sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session = Ort::Session(env, modelFilepath.c_str(), sessionOptions);

    
    Ort::MemoryInfo memoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    std::vector<float> input_data(1 * 3 * net_h * net_w);
    std::vector<float> output_data(1 * 3 * net_h * net_w);

    const std::vector<int64_t> input_shapes{1, 3, net_h, net_w};
    const std::vector<int64_t> output_shapes{1, 3, net_h, net_w};

    // create input output tenspr
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, input_data.data(), input_data.size(), input_shapes.data(), input_shapes.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memoryInfo, output_data.data(), output_data.size(), output_shapes.data(), output_shapes.size());

    
    // Prepare Input node
    size_t m_numInputs = session.GetInputCount();
    std::vector<const char*> input_node_names(m_numInputs);
    for (size_t i = 0; i < m_numInputs; i++) {
        char* input_name = session.GetInputNameAllocated(i, allocator).get();
        input_node_names[i] = input_name;
    }
    // Print input node names
    std::cout << "Input node names: " << std::endl;
    for (const auto& name : input_node_names) {
        std::cout << name << std::endl;
    }
    std::cout << "num Input:" << m_numInputs << std::endl;
    // Prepare Output node t
    size_t m_numOutputs = session.GetOutputCount();
    std::vector<const char*> output_node_names(m_numOutputs);
    for (size_t i = 0; i < m_numOutputs; i++) {
        char* output_name = session.GetOutputNameAllocated(i, allocator).get();
        output_node_names[i] = output_name;
    }
    std::cout << "Output node names: " << std::endl;
    for (const auto& oname : output_node_names) {
        std::cout << oname << std::endl;
    }
    std::cout << "num Output:" << m_numOutputs << std::endl;
    
    
    // Inference 
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, m_numInputs, output_node_names.data(), &output_tensor, m_numInputs);

}
/*
cv::Mat Inference::run_inference(const cv::Mat& input_image) {
    // Preprocess image
    cv::Mat img_resized;
    cv::resize(input_image, img_resized, cv::Size(net_w, net_h));
    img_resized.convertTo(img_resized, CV_32F, 1.0 / 255);
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

    // Create input tensor
    std::vector<float> input_tensor_values((float*)img_resized.data, (float*)img_resized.data + img_resized.total() * img_resized.channels());
    std::array<int64_t, 4> input_shape{1, 3, net_h, net_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    auto output_data = output_tensors.front().GetTensorMutableData<float>();

    // Create output depth map
    cv::Mat depth_map(net_h, net_w, CV_32F, output_data);
    return depth_map.clone();  // Clone to prevent shared memory issues
}

 */
