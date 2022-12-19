#include "src/fastertransformer/utils/debug.h"

namespace fastertransformer {

template<typename T>
void DumpData(const char* file_name, T* src_data, size_t size, bool device,char split)
{

    std::ostringstream oss;
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    T* host_temp = nullptr;
    if (device) {
        host_temp=new T[size];
        cudaMemcpy(host_temp, src_data, sizeof(T) * size, cudaMemcpyDeviceToHost);
    }
    else {
        host_temp=src_data;
    }

    for(size_t i=0;i<size;i++)
    {
      oss<<host_temp[i]<<split;
    }
    oss<<"\n";
    ofs.write(oss.str().c_str(), oss.str().length());
    ofs.close();
}

template<typename T>
void DumpHostData(const char* file_name, T* src_data, size_t size,char split)
{
    DumpData(file_name,src_data,size,false,split);
} 

template<typename T>
void DumpDeviceData(const char* file_name, T* src_data, size_t size,char split)
{
    DumpData(file_name,src_data,size,true,split);
} 



template void DumpDeviceData(const char* file_name, float* src_data, size_t size,char split);
template void DumpDeviceData(const char* file_name, half* src_data, size_t size,char split);
template void DumpDeviceData(const char* file_name, int* src_data, size_t size,char split);

template void DumpHostData(const char* file_name, float* src_data, size_t size,char split);
template void DumpHostData(const char* file_name, half* src_data, size_t size,char split);
template void DumpHostData(const char* file_name, int* src_data, size_t size,char split);

template void DumpData(const char* file_name, float* src_data, size_t size, bool device,char split)
template void DumpData(const char* file_name, half* src_data, size_t size, bool device,char split)
template void DumpData(const char* file_name, int* src_data, size_t size, bool device,char split)


#ifdef ENABLE_BF16
template void DumpDeviceData(const char* file_name, __nv_bfloat16* src_data, size_t size,char split);
template void DumpHostData(const char* file_name, __nv_bfloat16* src_data, size_t size,char split);
template void DumpData(const char* file_name, __nv_bfloat16* src_data, size_t size, bool device,char split)
#endif

}  // namespace fastertransformer