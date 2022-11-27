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
modify_cmd="python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --postemb --dymshape --eln --ln"

rst=`eval $modify_cmd | grep "Save modified onnx model to"`
rst=($rst)
idx=$((${#rst[@]}-1))
onnx_path=${rst[$idx]}
echo $onnx_path


# Build TensorRT Engine
# cd $ROOT_DIR
rm *.plan
./bin/onnx2trt $onnx_path ./Ernie_fp16.plan ./so/plugins/ --postemb --dymshape --fp16
