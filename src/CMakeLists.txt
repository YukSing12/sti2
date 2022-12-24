add_executable(static_trt_infer static_trt_infer.cpp)
add_executable(static_onnx2trt static_onnx2trt.cpp)


add_executable(dynamic_trt_infer dynamic_trt_infer.cpp)
add_executable(dynamic_onnx2trt dynamic_onnx2trt.cpp)

add_executable(ft_trt_infer ft_trt_infer.cpp)
add_executable(ft_onnx2trt ft_onnx2trt.cpp)


target_link_libraries(static_trt_infer cudart nvinfer nvinfer_plugin   dl)
target_link_libraries(static_onnx2trt cudart nvinfer nvinfer_plugin nvonnxparser  dl)

target_link_libraries(dynamic_trt_infer cudart nvinfer nvinfer_plugin   dl)
target_link_libraries(dynamic_onnx2trt cudart nvinfer nvinfer_plugin nvonnxparser  dl)

target_link_libraries(ft_trt_infer cudart nvinfer nvinfer_plugin   dl)
target_link_libraries(ft_onnx2trt cudart nvinfer nvinfer_plugin nvonnxparser  dl)



install(TARGETS static_trt_infer static_onnx2trt dynamic_trt_infer dynamic_onnx2trt  ft_trt_infer ft_onnx2trt    
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)



