#pragma once
#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include "memory_utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fastertransformer {

template<typename T>
void DumpData(const char* file_name, T* src_data, size_t size, bool device, char split);

template<typename T>
void DumpHostData(const char* file_name, T* src_data, size_t size, char split);

template<typename T>
void DumpDeviceData(const char* file_name, T* src_data, size_t size, char split);
}  // namespace fastertransformer