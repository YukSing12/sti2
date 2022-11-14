# 数据集准备

## 下载数据
从[官网](https://aistudio.baidu.com/aistudio/datasetdetail/174570)下载数据集并放置到指定目录

目录结构为
```
sti2_data/
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

## 复制到指定目录
```
cd sti2_data/data/*.txt data/
cp sti2_data/model/onnx_infer_model/model.onnx model
```
