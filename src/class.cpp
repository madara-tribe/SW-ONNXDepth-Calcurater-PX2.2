#include "depth_utility.h"

Inference::Inference(const std::string& model_path, bool useCUDA) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Inference");
    std::string modelFilepath{model_path};
     
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    // input 
    const auto inputName = session.GetInputNameAllocated(0, allocator);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    // output
    const auto outputName = session.GetOutputNameAllocated(0, allocator);
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    
     
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
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    auto output_data = output_tensors.front().GetTensorMutableData<float>();

    // Create output depth map
    cv::Mat depth_map(net_h, net_w, CV_32F, output_data);
    return depth_map.clone();  // Clone to prevent shared memory issues
}

