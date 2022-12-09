add_executable(main main.cpp)
add_executable(main_ft main_ft.cpp)
add_executable(main_ftv3 main_ftv3.cpp)
add_executable(onnx2trt onnx2trt.cpp)
add_executable(onnx2trt_multiprofile onnx2trt_multiprofile.cpp)
add_executable(onnx2trt_ft onnx2trt_ft.cpp)
add_executable(multiprofile multiprofile.cpp)
add_executable(ms_ft ms_ft.cpp)

target_link_libraries(main cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(main_ft cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(main_ftv3 cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(onnx2trt cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(onnx2trt_multiprofile cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(onnx2trt_ft cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(multiprofile cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(ms_ft cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)

install(TARGETS main main_ft onnx2trt onnx2trt_ft onnx2trt_multiprofile multiprofile ms_ft main_ftv3
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)



