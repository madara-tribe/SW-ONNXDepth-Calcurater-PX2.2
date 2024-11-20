cmake_minimum_required(VERSION 3.13)

set(LIBRARY_NAME ${PROJECT_NAME})

add_executable(inference inference.cpp midas_inference.cpp)
# include
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# onnxruntime
target_include_directories(${LIBRARY_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${ONNX_RUNTIME_LIB})

# cv2 
target_include_directories(${LIBRARY_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCV_LIBS})
