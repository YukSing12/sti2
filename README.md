# STI2

## 介绍
百度搜索技术创新挑战赛-赛道2

## 软件架构
软件架构说明
```
sti2/
|-- Ernie.plan
|-- LICENSE
|-- README.md
|-- bin    
|-- build.sh
|-- data
|-- label.res.txt
|-- model
|-- onnx2trt.py
|-- perf.res.txt
|-- requirements.txt
|-- run.sh
|-- so
|-- src
`-- test.sh
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
bash test.sh
```
