set(addrelu_files
        AddReluPlugin.cu
)

set(LIB_NAME "AddReluPlugin")
add_library(${LIB_NAME} SHARED ${addrelu_files})
set_target_properties(${LIB_NAME} PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${LIB_NAME} -lcudart -lnvinfer)

install(TARGETS ${LIB_NAME}
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/so/plugins)