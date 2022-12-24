# STI2

## 介绍
百度搜索技术创新挑战赛-赛道2

## 目录架构
目录架构说明
```
sti2/
|-- bin
|-- data
|-- include
|   |-- plugins
|   `-- tensorrt
|       `-- include 
|-- model
|-- so
|   |-- plugins
|   `-- tensorrt
|       `-- lib 
|-- src
|   |-- ft_ernie
|   |-- plugins
|   `-- python
|-- tests
|   |-- plugins
|   `-- testAllPlugins.sh
|-- tools
|-- run.sh
|-- build.sh
|-- test.sh
|-- requirements.txt
|-- CMakeLists.txt
|-- README.md
`-- LICENSE
```


## 安装教程

1.  克隆项目
```
git clone https://gitee.com/YukSing12/sti2.git
```
2.  从[官网](https://aistudio.baidu.com/aistudio/competition/detail/674/0/introduction)下载数据集
3.  解压数据集，将数据集下model.onnx  复制到  sti2/model下
4.  准备TensorRT环境(准备NV NGC TensorRT Docker 或者使用AIStudio  基于tensorRT 8.5.1.7 Cuda11.2  cuDnn 8.2)
```
cd sti2
ln -s /[tensorrt dir]/include sti2/include/tensorrt/include
ln -s /[tensorrt dir]/lib sti2/so/tensorrt/lib
or
cp -r /[tensorrt dir]/include sti2/include/tensorrt/include
cp -r /[tensorrt dir]/lib sti2/so/tensorrt/lib
```
5.  安装依赖
```
pip install -r requirements.txt
```

## 使用说明

1.  从paddlepaddle模型导出成onnx模型(可选)
2.  构建trt引擎、编译可执行文件
```
bash build.sh
```
3.  运行推理
```
bash run.sh data/label.test.txt
bash run.sh data/perf.test.txt
```
4. 本地测试
```
bash test.sh [exe]
eg:bash test.sh ft_infer
```
## 项目特性
### （一）部分算子融合推理
- 说明：融合算子分为两部分，
- 1. 改变输入的算子融合,有POSTEMB和DYNAMIC SHAPE两种。
- 2. 不改变输入的算子融合,有LayernormPlugin、AddResidualLayernormPlugin、EmbLayernormPlugin。</br>
- 该推理使用static_trt_infer.cpp 和 static_onnx2trt.cpp 
### （二）优化推理机制
- 说明: 此方式采用了multiprofile和cuda graph技术，仅支持第二维为dynamic shape的情况
- 该推理使用dynamic_trt_infer.cpp 和 dynamic_onnx2trt.cpp 
### （三）使用Faster Transformer
- 说明: 此方式采用了英伟达Faster Transformer框架，分为ft tensorRT Plugin和C++两种推理方式，本项目默认采用C++的方式推理
- ft_trt_infer.cpp 和 ft_onnx2trt.cpp为将Faster Transformer编译为tensorRT的plugin方式进行推理
- ft_ernie为C++方式推理，详见[src/ft_ernie]
### （四）使用方法
- 使用不同的推理方式。需要修改build.sh,详见 build.sh
