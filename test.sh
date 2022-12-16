ROOT_DIR=$(cd $(dirname $0); pwd)
export LD_LIBRARY_PATH=$ROOT_DIR/so/tensorrt/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD="$ROOT_DIR/so/tensorrt/lib/libnvinfer_builder_resource.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer_plugin.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvinfer.so.8.5.1  $ROOT_DIR/so/tensorrt/lib/libnvonnxparser.so.8.5.1 $ROOT_DIR/so/tensorrt/lib/libnvparsers.so.8.5.1"

if [ $# != 1 ]; then
    echo -e "Usage test.sh <exe>"
    echo -e "  <exe>  Execute program for testing."
    echo -e "         Supported exe: main: static shape inference using TensorRT"
    echo -e "                        main_ft: dynamic shape inference using FasterTransformer and TensorRT"
    echo -e "                        ernie_infer: dynamic shape inference using FasterTransformer"
    echo -e "                        multiprofile: dynamic shape and multiprofile inference using TensorRT"
    exit 1
fi

rm *.res.txt -f
# rm gemm_config*.in
if [ "$1" == "ernie_infer" ];then
    ./bin/ernie_infer 1 ./model/bin ./data/label.test.txt ./label.res.txt --useCudaGraph
    ./bin/ernie_infer 1 ./model/bin ./data/perf.test.txt ./perf.res.txt --useCudaGraph
else
    ./bin/$1 ./model/Ernie.plan ./data/label.test.txt ./label.res.txt ./so/plugins
    ./bin/$1 ./model/Ernie.plan ./data/perf.test.txt ./perf.res.txt ./so/plugins
fi
nan_num=`cat label.res.txt | grep nan | wc -l`
echo "Found $nan_num nan in label.res.txt"
nan_num=`cat perf.res.txt | grep nan | wc -l`
echo "Found $nan_num nan in perf.res.txt"

/usr/bin/python3 <<-EOF
import numpy as np
delta=0.01
with open("data/perf.res.txt", 'r') as fid1,\
        open("perf.res.txt", 'r') as fid2:
    lines1 = fid1.readlines()
    lines2 = fid2.readlines()
    assert (len(lines1) == len(lines2))
    mean_diff = 0
    max_diff = 0
    count = 0
    large_diff_count=0
    for i in range(len(lines1)):
        line1 = lines1[i]
        line2 = lines2[i]
        data1 = np.array([float(x) for x in line1.split("\t")[2].split(",")])
        data2 = np.array([float(x) for x in line2.split("\t")[2].split(",")])
        abs_diff = np.abs(data1 - data2)
        large_diff_count+=np.sum(abs_diff>delta)
        if(np.sum(abs_diff>delta)>0.01):
            print("qid:{} might error".format(line2.split('\t')[0]))
        max_diff = max_diff if max_diff > abs_diff.max() else abs_diff.max()
        mean_diff = np.sum(abs_diff)
        count += len(abs_diff)
    
    mean_diff /= count
    print("Max diff is {}".format(max_diff))
    print("diff > {} count is {}".format(delta,large_diff_count))
    print("Mean diff is {}".format(mean_diff))
EOF

python tools/local_evaluate.py ./label.res.txt
python tools/local_evaluate.py ./perf.res.txt