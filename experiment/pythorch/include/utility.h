#include <opencv2/opencv.hpp>
#include <iostream>

void image_info(cv::Mat image){
    double min, max;
    std::cout << "H: " << image.rows << " W: " << image.cols << ", channels: " << image.cols << std::endl;
    cv::minMaxLoc(image, &min, &max);
    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;
}
