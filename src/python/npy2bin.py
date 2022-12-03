import os
import sys
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser('Extract Weights of Ernie', add_help=False)
    parser.add_argument('--npy', required=True, type=str, help='Path of npy file to load')
    parser.add_argument('--bin', default='./model/bin', type=str, help='Path of bin')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    os.system("mkdir -p "+ args.bin)

    ews = np.load(args.npy, allow_pickle='TRUE')
    ews = ews.item()

    for name in ews:
        saved_path = os.path.join(args.bin, name+".bin")
        cur = ews[name]
        # if name.endswith(".weight") and len(cur.shape)==2:
        #     cur = cur.transpose((1,0))
        #     #print(name, cur.shape)
        cur.astype(np.float32).tofile(saved_path)


    print("Extract weights of Ernie finish!")

