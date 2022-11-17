ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

# Build so
bash build_plugins.sh

# Build cpp
cd $ROOT_DIR/src
make clean
make

# Modify TensorRT Engine
cd $ROOT_DIR
python modify_ERNIE.py

# Build TensorRT Engine
cd $ROOT_DIR
python onnx2trt.py





