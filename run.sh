ROOT_DIR=$(cd $(dirname $0); pwd)
 
export LD_LIBRARY_PATH=$ROOT_DIR/so/tensorrt/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD="$ROOT_DIR/so/tensorrt/lib/libnvinfer_builder_resource.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer_plugin.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvonnxparser.so.8.5.1 $ROOT_DIR/so/tensorrt/lib/libnvparsers.so.8.5.1"

if [ $1 = "data/perf.test.txt" ]; then
    rm  ./perf.res.txt
    ./bin/main ./model/Ernie.plan ./data/perf.test.txt ./perf.res.txt ./so/plugins/
else
     rm  ./label.res.txt
    ./bin/main ./model/Ernie.plan ./data/label.test.txt ./label.res.txt ./so/plugins/
fi
