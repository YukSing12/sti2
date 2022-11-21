ROOT_DIR=$(cd $(dirname $0); pwd)
 
perf_mode=`echo $1 | grep "perf" | wc -l`
label_mode=`echo $1 | grep "label" | wc -l`

if [[ $perf_mode == 1 ]]; then
    rm  ./perf.res.txt
    ./bin/main ./Ernie_fp16.plan ./data/perf.test.txt ./perf.res.txt ./so/plugins/
else
     rm  ./label.res.txt
    ./bin/main ./Ernie_fp16.plan ./data/label.test.txt ./label.res.txt ./so/plugins/
fi
