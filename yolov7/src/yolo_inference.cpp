#include "utility.h"
#include "yolo_inference.h"

#include <iostream>
#include <cstring>

YoloDetect::YoloDetect(const std::string& modelPath){
    session = new Ort::Session(env, modelPath, session_options);
    
}
YoloDetect::~YoloDetect() {
    delete session;
}

float* YoloDetect::RunSession(cv::Mat inputImage){
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<int64_t> inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    model_input_height = inputDims.at(3);
    model_input_width = inputDims.at(2);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                              inputImage.ptr<float>(),
                                                              inputImage.total() * sizeof(float),
                                                              inputDims.data(),
                                                              inputDims.size());

    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{nullptr},
                                                        &input_node_name,
                                                        &inputTensor,
                                                        num_input_nodes,
                                                        &output_node_name,
                                                        num_output_nodes);
    float* rawOutput = outputTensors[0].GetTensorData<float>();
    return rawOutput
}
cv::Mat YoloDetect::preprocess(cv::Mat& image){

    // Channels order: BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Calculate the scaling factor for resizing without distortion
    double scale;
    if (image.cols / static_cast<double>(image.rows) > model_input_width / static_cast<double>(model_input_height)) {
        scale = model_input_width / static_cast<double>(image.cols);
    } else {
        scale = model_input_height / static_cast<double>(image.rows);
    }

    // Resize the image with keeping the aspect ratio
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(), scale, scale);

    model_height_after_padding = resizedImage.size[0];
    model_width_after_padding = resizedImage.size[1];
    // Create a blank canvas with the desired model input size
    cv::Mat paddedImage = cv::Mat::zeros(model_input_height, model_input_width, resizedImage.type());

    // Calculate the position to paste the resized image
    int x_offset = (paddedImage.cols - resizedImage.cols) / 2;
    int y_offset = (paddedImage.rows - resizedImage.rows) / 2;
    pad_size_y = y_offset;
    pad_size_x = x_offset;

    // Copy the resized image to the center of the canvas
    resizedImage.copyTo(paddedImage(cv::Rect(x_offset, y_offset, resizedImage.cols, resizedImage.rows)));
    // Convert image to float32 and normalize
    cv::Mat floatImage;
    paddedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Create a 4-dimensional blob from the image
    cv::Mat blobImage = cv::dnn::blobFromImage(floatImage);


    return blobImage;
}

std::vector<Result> YoloDetect::postprocess(cv::Size originalImageSize, float* rawOutput)
{
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    std::vector<Result> resultVector;

    for (int i = 0; i < outputShape[0]; i++) {

        float confidence        = output[i * outputShape[1] + 0];
        float x1                = output[i * outputShape[1] + 1];
        float y1                = output[i * outputShape[1] + 2];
        float x2                = output[i * outputShape[1] + 3];
        float y2                = output[i * outputShape[1] + 4];
        int classPrediction     = output[i * outputShape[1] + 5];
        float accuracy          = output[i * outputShape[1] + 6];

        (void) confidence;

        std::cout << "Class Name: " << classNames.at(classPrediction) << std::endl;
        std::cout << "Coords: Top Left (" << x1 << ", " << y1 << "), Bottom Right (" << x2 << ", " << y2 << ")" << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;

        // Coords should be scaled to the original image. The coords from the model are relative to the model's input height and width.
        x1 = ((x1- pad_size_x)  / model_width_after_padding) * originalImageSize.width ;
        x2 = ((x2- pad_size_x) / model_width_after_padding) * originalImageSize.width ;
        y1 = ((y1 - pad_size_y) / model_height_after_padding) * originalImageSize.height ;
        y2 = ((y2 - pad_size_y) / model_height_after_padding) * originalImageSize.height ;

        Result result( x1, x2, y1, y2, classPrediction, accuracy);

        resultVector.push_back( result );

        std::cout << std::endl;
    }

    return resultVector;
}

void YoloDetect::drawBoundingBox(cv::Mat& image, std::vector<Result>& resultVector){
    for( auto result : resultVector ) {

        if( result.accuracy > 0.6 ) { // Threshold, can be made function parameter

            cv::rectangle(image, cv::Point(result.x1, result.y1), cv::Point(result.x2, result.y2), cv::Scalar(0, 255, 0), 2);

            cv::putText(image, classNames.at( result.obj_id ),
                        cv::Point(result.x1, result.y1 - 3), cv::FONT_ITALIC,
                        0.8, cv::Scalar(255, 255, 255), 2);

            cv::putText(image, std::to_string(result.accuracy),
                        cv::Point(result.x1, result.y1+30), cv::FONT_ITALIC,
                        0.8, cv::Scalar(255, 255, 0), 2);
        }
    }

}
