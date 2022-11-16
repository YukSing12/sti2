ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

# Build cpp
cd $ROOT_DIR/src
make clean
make

# Build so
plugins="
    LayerNormPlugin
"

for plugin in $plugins
do
    echo "========================= Start building $plugin ========================"
    plugin_dir="$ROOT_DIR/src/${plugin}"
    cd $plugin_dir
    make clean
    make all
    cp "$plugin.so" $ROOT_DIR/so
done

# Modify TensorRT Engine
cd $ROOT_DIR/model
python modify_ERNIE.py

# Build TensorRT Engine
cd $ROOT_DIR
python onnx2trt.py





