clear
CURRENT_DIR=$(dirname $(readlink -f $0))
PROJECT_DIR=$CURRENT_DIR/
# model info
model_dir="$PROJECT_DIR/model/paddle_infer_model/"
model_filename='__model__'
params_filename='__params__'

echo "----------------Quantizing Ernie with PPQ---------------"
paddle2onnx --model_dir $model_dir --model_filename $model_filename --params_filename $params_filename --save_file model/model_13.onnx --opset_version 13
python -m paddle2onnx.optimize --input_model model/model_13.onnx --output_model model/model_dymshape_13.onnx --input_shape_dict "{'read_file_0.tmp_0': [-1, -1, 1], 'read_file_0.tmp_1': [-1, -1, 1], 'read_file_0.tmp_2': [-1, -1, 1], 'read_file_0.tmp_3': [-1, -1, 1], 'read_file_0.tmp_6': [-1, 1, 1], 'read_file_0.tmp_7': [-1, 1, 1], 'read_file_0.tmp_8': [-1, 1, 1], 'read_file_0.tmp_9': [-1, 1, 1], 'read_file_0.tmp_10': [-1, 1, 1], 'read_file_0.tmp_11': [-1, 1, 1], 'read_file_0.tmp_12': [-1, 1, 1], 'read_file_0.tmp_13': [-1, 1, 1]}"
# ppq only supports onnx opset<=13
python src/python/ppq_quantize.py

echo "----------------Exporting weights of Ernie in FP ---------------"
python src/python/onnx2bin.py --onnx model/model.onnx --bin model/bin

echo "----------------Exporting scale list of Ernie---------------"
# only export quantized weights of encoder
python src/python/qtonnx2bin.py --onnx model/qt_model_dymshape.onnx --bin model/bin
