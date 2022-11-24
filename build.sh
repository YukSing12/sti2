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
# python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --ln --postemb
python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx   --eln --postemb  --dymshape

# # Build TensorRT Engine
# # cd $ROOT_DIR
rm *.plan
python src/python/onnx2trt.py --eln --postemb --fp16 --dymshape
