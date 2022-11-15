ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

# Modify TensorRT Engine

# Build TensorRT Engine
python onnx2trt.py

# Build cpp
cd $ROOT_DIR/src
make clean
make

# Build so

