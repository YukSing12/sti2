ROOT_DIR=$(cd $(dirname $0); pwd)
echo $ROOT_DIR

# Build so
bash build_plugins.sh

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
    cp "$plugin.so" $ROOT_DIR/so/plugins
done

# Modify TensorRT Engine
cd $ROOT_DIR
python modify_ERNIE.py

# Build TensorRT Engine
cd $ROOT_DIR
python onnx2trt.py





