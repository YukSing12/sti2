# STI2

## 介绍
百度搜索技术创新挑战赛-赛道2

## 软件架构
软件架构说明
```
sti2/
├── app
│   └── app.cpp
├── LICENSE
├── onnx2trt
│   └── onnx2trt.py
├── pp2onnx
│   └── pp2onnx.py
├── README.md
└── sti2_data
    ├── data
    │   ├── label.test.txt
    │   └── perf.test.txt
    ├── example.label.res.txt
    ├── example.perf.res.txt
    ├── local_evaluate.py
    ├── model
    │   ├── onnx_infer_model
    │   │   └── model.onnx
    │   └── paddle_infer_model
    │       ├── __model__
    │       └── __params__
    └── README.md
```


## 安装教程

1.  克隆项目
```
git clone https://gitee.com/YukSing12/sti2.git
```
2.  下载数据集，详见[文档](./sti2_data/README.md)
3.  准备docker(使用AIStudio 环境)
4.  安装依赖
```
pip install -r requirements.txt
```

## 使用说明

1.  从paddlepaddle模型导出成onnx模型(可选)
2.  从onnx模型导出成trt引擎
```
python onnx2trt.py
```
3.  编译并运行可执行文件
