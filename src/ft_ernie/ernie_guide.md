# Guide for Ernie

## 导出模型权重
```
python src/python/onnx2torch.py --onnx model/model.onnx --npy model/model.npy
python src/python/npy2bin.py --npy model/model.npy --bin model/bin/
```

## 编译fastertransformer
```
cd src/ft_ernie
bash compile.sh
```

设置脚本里的选项为Release还是Debug
```
-DCMAKE_BUILD_TYPE=Debug
```

## 构建模型
```
cd sti2
bash build.sh
```

build.sh中,开启--ft来替换onnx图
```
python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --postemb --ft
```

## 运行模型
```
bash test.sh main_ft
```

## 目录结构
```
|-- 3rdparty
|-- CMakeLists.txt
|-- benchmarks
|-- cmake
|-- compile.sh
|-- docker
|-- docs
|-- ernie_guide.md
|-- examples
|-- src
|   |-- CMakeLists.txt
|   `-- fastertransformer
|-- CMakeLists.txt
|-- kernels
|   |-- CMakeLists.txt
|   |-- ernie_kernels.cu (ernie的kernels,例如预处理的embedding)
|   `-- ernie_kernels.h
|-- layers
|-- models
|   |-- BaseWeight.h
|   |-- CMakeLists.txt
|   |-- ernie
|   |   |-- CMakeLists.txt
|   |   |-- ErnieEncoder.cc             (ernie的网络的结构初始化及执行逻辑)
|   |   |-- ErnieEncoder.h
|   |   |-- ErnieEncoderLayerWeight.cc  (ernie的MHA权重加载)
|   |   |-- ErnieEncoderLayerWeight.h
|   |   |-- ErnieEncoderWeight.cc       (ernie的整体权重加载,会加载MHA权重,及embedding表,pre_layernorm权重等)
|   |   |-- ErnieEncoderWeight.h
|   |   `-- ernie_gemm.cc
|-- tensorrt_plugin
|   |-- CMakeLists.txt
|   |-- ernie
|   |   |-- CMakeLists.txt
|   |   |-- ErniePlugin.cu              (ernie plugin的封装, 实际上调用ErnieEncoder的forward())
|   |   |-- ErniePlugin.h
|   |   |-- ErniePluginGemm.cc
|   |   |-- ErniePluginGemm.h
|   |   `-- README.md
|-- tf_op
|-- th_op
|-- triton_backend
`-- utils
    |-- gemm_test
    |   |-- CMakeLists.txt
    |   |-- ernie_gemm_func.cc
    |   `-- ernie_gemm_func.h
|-- templates
`-- tests
```