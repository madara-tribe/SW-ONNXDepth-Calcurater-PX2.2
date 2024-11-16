#include <iostream>
#include <fstream>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "utility.h"
#include "depth_utility.h"

int model_check(const std::string& onnx_model_path) {
    // Initialize the ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Model");
    std::string modelFilepath{onnx_model_path};
    std::ifstream modelFile(modelFilepath);
    if (!modelFile.good()) {
        std::cerr << "Model file does not exist or could not be found: " << modelFilepath << std::endl;
    return -1;
    }
    // Create an ONNX Runtime session options object
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Load the model
    std::cout << "model name " << modelFilepath.c_str() << std::endl;
    Ort::Session session(env, modelFilepath.c_str(), session_options);
    // Get input and output details
    size_t num_input_nodes = session.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;

    for (size_t i = 0; i < num_input_nodes; i++) {
        // Updated to use GetInputNameAllocated
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << ": " << input_name.get() << std::endl;
    }
    std::cout << "Model loaded successfully!" << std::endl;
    return 0;
}


void draw_depth(const std::vector<float>& depth_map, int w, int h) {
    float min_depth = *std::min_element(depth_map.begin(), depth_map.end());
    float max_depth = *std::max_element(depth_map.begin(), depth_map.end());

    // Normalize depth map
    cv::Mat depth_image(h, w, CV_32F);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            depth_image.at<float>(i, j) = (depth_map[i * w + j] - min_depth) / (max_depth - min_depth);
        }
    }

    // Convert to 8-bit and apply color map
    cv::Mat depth_8u;
    depth_image.convertTo(depth_8u, CV_8U, 255);
    cv::Mat depth_colored;
    cv::applyColorMap(depth_8u, depth_colored, cv::COLORMAP_JET);

    // Save the depth image
    cv::imwrite("colored_depth.png", depth_colored);
}
