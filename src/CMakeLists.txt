add_executable(main main.cpp)
add_executable(onnx2trt onnx2trt.cpp)
add_executable(pinned_graph pinned_graph.cpp)
add_executable(multi_profile multi_profile.cpp)
add_executable(multi_profile_infer multi_profile_infer.cpp)

target_link_libraries(main cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(onnx2trt cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(pinned_graph cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(multi_profile cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)
target_link_libraries(multi_profile_infer cudart nvinfer nvinfer_plugin nvonnxparser nvonnxparser dl)


install(TARGETS main onnx2trt pinned_graph multi_profile  multi_profile_infer
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)



