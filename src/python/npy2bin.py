import os
import sys
import argparse
import numpy as np
import configparser

def get_args():
    parser = argparse.ArgumentParser(
        'Extract Weights of Ernie', add_help=False)
    parser.add_argument('--npy', required=True, type=str,
                        help='Path of npy file to load')
    parser.add_argument('--bin', default='./model/bin',
                        type=str, help='Path of bin')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 mode or not, default is FP32')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    os.system("mkdir -p " + args.bin)

    conf = configparser.ConfigParser()
    
    conf.add_section("ernie")
    with open(os.path.join(args.bin, "config.ini"), 'w') as fid:
        if args.fp16:
            print("Extract weights in FP16 mode")
            conf.set("ernie", "weight_data_type", "fp16")
            npDataType = np.float16
        else:
            print("Extract weights in FP32 mode")
            conf.set("ernie", "weight_data_type", "fp32")
            npDataType = np.float32
        conf.write(fid)
    
    ews = np.load(args.npy, allow_pickle='TRUE')
    ews = ews.item()

    for name in ews:
        saved_path = os.path.join(args.bin, name+".bin")
        cur = ews[name]
        print(name, cur.shape)
        cur.astype(npDataType).tofile(saved_path)

    print("Succeed extracting weights of Ernie!")
