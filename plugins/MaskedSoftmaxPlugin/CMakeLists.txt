add_library(MaskedSoftmaxPlugin SHARED MaskedSoftmaxPlugin.cu)
target_link_libraries(MaskedSoftmaxPlugin nvinfer)
install(TARGETS MaskedSoftmaxPlugin
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/so)
