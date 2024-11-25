cmake_minimum_required(VERSION 3.13)

set(LIBRARY_NAME ${PROJECT_NAME})

add_executable(inference inference.cpp) #yolo_inference.cpp)

# Include directories
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link libraries (onnxruntime opencv pytorch) 
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS})

# C++ standard
set_property(TARGET ${LIBRARY_NAME} PROPERTY CXX_STANDARD 17)

# Include ONNX Runtime directories
target_include_directories(${LIBRARY_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})


