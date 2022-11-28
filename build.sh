ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

export LD_LIBRARY_PATH=$ROOT_DIR/so/tensorrt/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD="$ROOT_DIR/so/tensorrt/lib/libnvinfer_builder_resource.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer_plugin.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvonnxparser.so.8.5.1 $ROOT_DIR/so/tensorrt/lib/libnvparsers.so.8.5.1"

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
rm model/*.plan
./bin/onnx2trt_multiprofile $onnx_path ./model/Ernie.plan ./so/plugins/ --postemb --dymshape --fp16
