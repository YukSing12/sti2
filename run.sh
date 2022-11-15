ROOT_DIR=$(cd $(dirname $0); pwd)

cd $ROOT_DIR/bin
perf_mode=`echo $1 | grep "perf" | wc -l`
label_mode=`echo $1 | grep "label" | wc -l`
if [[ $perf_mode == 1 ]]; then
    ./main.exe perf
fi
if [[ $label_mode == 1 ]]; then
    ./main.exe label
fi

