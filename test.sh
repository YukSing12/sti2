ROOT_DIR=$(cd $(dirname $0); pwd)

echo "========================== Evaluate performance of model ========================"
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=Ernie.plan \
    --profilingVerbosity=detailed \
    --minShapes=read_file_0.tmp_0:1x128x1,read_file_0.tmp_1:1x128x1,read_file_0.tmp_2:1x128x1,read_file_0.tmp_3:1x128x1,read_file_0.tmp_6:1x1x1,read_file_0.tmp_7:1x1x1,read_file_0.tmp_8:1x1x1,read_file_0.tmp_9:1x1x1,read_file_0.tmp_10:1x1x1,read_file_0.tmp_11:1x1x1,read_file_0.tmp_12:1x1x1,read_file_0.tmp_13:1x1x1 \
    --optShapes=read_file_0.tmp_0:4x128x1,read_file_0.tmp_1:4x128x1,read_file_0.tmp_2:4x128x1,read_file_0.tmp_3:4x128x1,read_file_0.tmp_6:4x1x1,read_file_0.tmp_7:4x1x1,read_file_0.tmp_8:4x1x1,read_file_0.tmp_9:4x1x1,read_file_0.tmp_10:4x1x1,read_file_0.tmp_11:4x1x1,read_file_0.tmp_12:4x1x1,read_file_0.tmp_13:4x1x1 \
    --maxShapes=read_file_0.tmp_0:10x128x1,read_file_0.tmp_1:10x128x1,read_file_0.tmp_2:10x128x1,read_file_0.tmp_3:10x128x1,read_file_0.tmp_6:10x1x1,read_file_0.tmp_7:10x1x1,read_file_0.tmp_8:10x1x1,read_file_0.tmp_9:10x1x1,read_file_0.tmp_10:10x1x1,read_file_0.tmp_11:10x1x1,read_file_0.tmp_12:10x1x1,read_file_0.tmp_13:10x1x1 \
    --plugins=./so/LayerNormPlugin.so
    # --plugins=./AddVBiasTransposePlugin.so \
    # --plugins=./LayerNormPlugin.so \
    # --plugins=./Mask2PosPlugin.so \
    # --plugins=./MaskedSoftmaxPlugin.so

echo "========================== Evaluate accuracy of model ========================"
python src/utils/local_evaluate.py ./label.res.txt
python src/utils/local_evaluate.py ./perf.res.txt