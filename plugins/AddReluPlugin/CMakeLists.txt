add_library(AddReluPlugin SHARED AddReluPlugin.cu)
target_link_libraries(AddReluPlugin nvinfer)
install(TARGETS AddReluPlugin
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/so/plugins)
