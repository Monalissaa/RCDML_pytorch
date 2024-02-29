import argparse
import sys
import numpy as np

import os, time
import math
import re
import tempfile
from tempfile import mkdtemp
from sklearn import preprocessing
from sklearn.decomposition import PCA
from subprocess import Popen, check_output
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize
import time

np.random.seed(42)


def main():
    for factor in [0.5, 1, 2, 0.01]:
        # for factor in [0.01, 0.02,0.1, 0.2, 0.3, 0.4]:
            # if (sigma == 2 and factor == 0.01) or (sigma == 2 and factor == 0.02) or (sigma == 2 and factor == 0.1) or (sigma == 2 and factor == 0.2) or (sigma == 2 and factor: 0.3):
            #     continue
        print(' factor: %s' % (factor))
        cmd1 = 'python /home/wangxinru/program/Proxy-Anchor-CVPR2020-master/code/train.py --gpu-id 0 ' \
               '--loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --factor %s' % (factor)
        # print(cmd1)
        check_output(cmd1, shell=True)



if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    time = time_end - time_start
    print('totally cost:{0} minutes'.format(round(time // 60.00, 2)))