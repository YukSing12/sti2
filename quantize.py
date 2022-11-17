import argparse
import paddle
import paddleslim

def quantize(model_dir, 
        model_filename,
        params_filename,
        model_dir_quant_dynamic,
        save_model_filename,
        save_params_filename,
        weight_bits,
        opt=None):
    # open static map
    paddle.enable_static()
    # 动态量化
    paddleslim.quant.quant_post_dynamic(
    model_dir=model_dir, # 输入模型路径
    model_filename=model_filename, # 输入模型计算图文件名称
    params_filename=params_filename, # 输入模型参数文件名称
    save_model_dir=model_dir_quant_dynamic, # 输出模型路径
    save_model_filename=save_model_filename, # 输出模型计算图名称
    save_params_filename=save_params_filename, # 输出模型参数文件名称
    weight_bits=8)

    print("-----------------suceess------------------")
    print("file save in ", model_dir_quant_dynamic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='model and params path')
    parser.add_argument('--model_filename', type=str, help='model filename')
    parser.add_argument('--params_filename', type=str, help='params filename')
    parser.add_argument('--model_dir_quant_dynamic', type=str, help='save filepath')
    parser.add_argument('--save_model_filename', type=str, default = '__model__quan', help='save model filename')
    parser.add_argument('--save_params_filename', type=str, default = '__params__quan', help='save params filename')
    parser.add_argument('--weight_bits', type=int, default = 8, help='quantize 8/16 to int8/16')
    
    opt = parser.parse_args()
    print(opt)

    quantize(opt.model_dir,
    opt.model_filename,
    opt.params_filename,
    opt.model_dir_quant_dynamic,
    opt.save_model_filename,
    opt.save_params_filename,
    opt.weight_bits,
    opt
    )