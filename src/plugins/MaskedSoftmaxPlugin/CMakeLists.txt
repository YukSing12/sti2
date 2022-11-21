add_library(MaskedSoftmaxPlugin SHARED MaskedSoftmaxPlugin.cu)
target_link_libraries(MaskedSoftmaxPlugin nvinfer nvinfer_plugin nvonnxparser)
install(TARGETS MaskedSoftmaxPlugin
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/so/plugins)
