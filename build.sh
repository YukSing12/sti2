ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

export LD_LIBRARY_PATH=$ROOT_DIR/so/tensorrt/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD="$ROOT_DIR/so/tensorrt/lib/libnvinfer_builder_resource.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer_plugin.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvonnxparser.so.8.5.1 $ROOT_DIR/so/tensorrt/lib/libnvparsers.so.8.5.1"

# Build Fastertransformer
cd $ROOT_DIR/src/ft_ernie/
bash compile.sh

# Export weights from onnx to bin
cd $ROOT_DIR
python src/python/onnx2bin.py --onnx model/model.onnx --bin model/bin

