# STI2

## 介绍
百度搜索技术创新挑战赛-赛道2

## 目录架构
目录架构说明
```
sti2/
|-- CMakeLists.txt
|-- LICENSE
|-- README.md
|-- bin
|-- build.sh
|-- data
|-- include
|   |-- plugins
|   `-- tensorrt
|       `-- include -> /opt/tensorrt/include/
|-- model
|-- requirements.txt
|-- run.sh
|-- so
|   |-- plugins
|   `-- tensorrt
|       `-- lib -> /opt/tensorrt/lib
|-- src
|   |-- plugins
|   `-- python
|-- test
|   |-- plugins
|   `-- test.sh
`-- tools
```


## 安装教程

1.  克隆项目
```
git clone https://gitee.com/YukSing12/sti2.git
```
2.  从[官网](https://aistudio.baidu.com/aistudio/competition/detail/674/0/introduction)下载数据集
3.  准备TensorRT环境(准备NV NGC TensorRT Docker 或者使用AIStudio)
```
cd sti2
ln -s /opt/tensorrt/include sti2/include/tensorrt/include
ln -s /opt/tensorrt/lib sti2/so/tensorrt/lib
```
4.  安装依赖
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
bash test/test.sh
```
