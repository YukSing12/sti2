CURRENT_DIR=$(dirname $(readlink -f $0))
# model info
model_dir="$CURRENT_DIR/model/paddle_infer_model/"
model_filename='__model__'
params_filename='__params__'
# output file
model_dir_quant_dynamic="$CURRENT_DIR/model/"
save_model_filename='model_quan'
save_params_filename='params_quan'
weight_bits=8

print_usages(){
    echo "Usages: run_one_stop.sh [q] $1
                  q: quantize"
    exit 1
}



run_quantize()
{
    echo "----------------Quantizing ${model}---------------"
    python quantize.py --model_dir $model_dir --model_filename $model_filename --params_filename $params_filename --model_dir_quant_dynamic $model_dir_quant_dynamic --save_model_filename $save_model_filename --save_params_filename $save_params_filename --weight_bits 8
}

clear
if [[ $# -lt 1 ]];then
    print_usages
fi
while getopts ":q" opt
do
    case $opt in
        q)
        run_quantize
        if [ $? != 0 ]; then
            echo "Error occurs in quantization."
            exit 1
        fi;;
        ?)
        print_usages;;    
    esac
done