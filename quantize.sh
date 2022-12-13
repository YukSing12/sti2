clear
CURRENT_DIR=$(dirname $(readlink -f $0))
PROJECT_DIR=$CURRENT_DIR/
# model info
model_dir="$PROJECT_DIR/model/paddle_infer_model/"
model_filename='__model__'
params_filename='__params__'
# output file
model_dir_quant_dynamic="$PROJECT_DIR/model/"
save_model_filename='model_quan'
save_params_filename='params_quan'

printHelp()
{
    echo -e "Usage: quantize.sh <tool>"
    echo -e "\t<tool>\tSupported quantization tool:"
    echo -e "         \tppq: OpenPPQ"
    echo -e "         \tppslim: PaddleSlim (Experimental method)"
    exit 1
}

if [ $# != 1 ]; then
    printHelp
fi

if [ "$1" == "ppq" ];then
    echo "----------------Quantizing Ernie with PPQ---------------"
    paddle2onnx --model_dir $model_dir --model_filename $model_filename --params_filename $params_filename --save_file model/model_13.onnx --opset_version 13
    python -m paddle2onnx.optimize --input_model model/model_13.onnx --output_model model/model_dymshape_13.onnx --input_shape_dict "{'read_file_0.tmp_0': [-1, -1, 1], 'read_file_0.tmp_1': [-1, -1, 1], 'read_file_0.tmp_2': [-1, -1, 1], 'read_file_0.tmp_3': [-1, -1, 1], 'read_file_0.tmp_6': [-1, 1, 1], 'read_file_0.tmp_7': [-1, 1, 1], 'read_file_0.tmp_8': [-1, 1, 1], 'read_file_0.tmp_9': [-1, 1, 1], 'read_file_0.tmp_10': [-1, 1, 1], 'read_file_0.tmp_11': [-1, 1, 1], 'read_file_0.tmp_12': [-1, 1, 1], 'read_file_0.tmp_13': [-1, 1, 1]}"
    # ppq only supports onnx opset<=13
    python src/python/ppq_quantize.py

    echo "----------------Exporting weights of Ernie in FP ---------------"
    python src/python/onnx2bin.py --onnx model/model.onnx --bin model/bin

    echo "----------------Exporting weights of Ernie in INT8---------------"
    # only export quantized weights of encoder
    python src/python/qtonnx2bin.py --onnx model/qt_model_dymshape.onnx --bin model/bin
elif [ "$1" == "ppslim" ];then
    echo "----------------Quantizing Ernie with PaddleSlim---------------"
    python src/python/ppslim_quantize.py --model_dir $model_dir --model_filename $model_filename --params_filename $params_filename --model_dir_quant_dynamic $model_dir_quant_dynamic --save_model_filename $save_model_filename --save_params_filename $save_params_filename --weight_bits 8
    echo "----------------Exporting weights of Ernie in FP ---------------"
    python src/python/onnx2bin.py --onnx model/model.onnx --bin model/bin
    echo "----------------Can not export weights of Ernie in INT8---------------"
else
    echo "Unsupported quantization tool: $1"
    printHelp
fi
