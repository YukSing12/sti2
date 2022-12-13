import argparse
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as reader

import paddleslim
from paddleslim.quant import quant_post_static
import numpy as np

BATCHSIZE = 1
SEQ_LEN = 128
INPUTS = [([BATCHSIZE, SEQ_LEN, 1], np.int64),
          ([BATCHSIZE, SEQ_LEN, 1], np.int64),
          ([BATCHSIZE, SEQ_LEN, 1], np.int64),
          ([BATCHSIZE, SEQ_LEN, 1], np.float32),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64),
          ([BATCHSIZE, 1, 1], np.int64)]

CAL_DATASET = 'data/label.test.txt'
def cal_reader(): 
    inputs_list = INPUTS
    cal_dataset_path = CAL_DATASET
    with open(cal_dataset_path, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            meta_data = line.split(";")
            qid = meta_data[0]
            label = meta_data[1]
            i = 2
            sample = []
            for name in range(len(inputs_list)):
                data = meta_data[i]
                tmp = data.split(":")
                shape = tuple(int(s) for s in tmp[0].split(" "))
                value = [inputs_list[name][1](v) for v in tmp[1].split(" ")]
                value = np.array(value).reshape(shape)
                sample.append(value)
                i += 1
            yield sample

def quantize(model_dir,
             model_filename,
             params_filename,
             quantize_model_path,
             save_model_filename,
             save_params_filename,
             weight_bits,
             opt=None):

    paddle.enable_static()

    place = paddle.CUDAPlace(
        0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    quant_post_static(
        executor=exe,
        model_dir=model_dir,
        quantize_model_path=quantize_model_path,
        sample_generator=cal_reader,
        model_filename=model_filename,
        params_filename=params_filename,
        batch_size=8,
        batch_nums=10,
        onnx_format=True)

    print("-----------------suceess------------------")
    print("file save in ", quantize_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='model and params path')
    parser.add_argument('--model_filename', type=str, help='model filename')
    parser.add_argument('--params_filename', type=str, help='params filename')
    parser.add_argument('--quantize_model_path',
                        type=str, help='save filepath')
    parser.add_argument('--save_model_filename', type=str,
                        default='__model__quan', help='save model filename')
    parser.add_argument('--save_params_filename', type=str,
                        default='__params__quan', help='save params filename')
    parser.add_argument('--weight_bits', type=int, default=8,
                        help='quantize 8/16 to int8/16')

    opt = parser.parse_args()

    quantize(opt.model_dir,
             opt.model_filename,
             opt.params_filename,
             opt.quantize_model_path,
             opt.save_model_filename,
             opt.save_params_filename,
             opt.weight_bits,
             opt
             )
