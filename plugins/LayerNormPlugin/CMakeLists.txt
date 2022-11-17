add_library(LayerNormPlugin SHARED LayerNormPlugin.cu)
target_link_libraries(LayerNormPlugin nvinfer)
install(TARGETS LayerNormPlugin
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/so/plugins)
