#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <torch/torch.h>

int main() {
  std::cout << "Pytorch Version: " << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
}
