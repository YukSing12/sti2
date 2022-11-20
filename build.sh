ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

# Build cpp and so
rm -rf build
mkdir build
cd build
cmake ..
make -j16
make install

cd $ROOT_DIR
# Modify TensorRT Engine
python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --ln

# Build TensorRT Engine
# cd $ROOT_DIR
python src/python/onnx2trt.py --onnx model/modified_model_ln.onnx --fp16





