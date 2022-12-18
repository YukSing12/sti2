#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fastertransformer {

template<typename T>
void DumpTxt(const char* file_name, T* src_data, size_t size, bool device,char split);

template<typename T>
void HostDumpTxt(const char* file_name, T* src_data, size_t size,char split);

template<typename T>
void DeviceDumpTxt(const char* file_name, T* src_data, size_t size,char split);
}