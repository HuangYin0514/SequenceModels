# -*- coding: utf-8 -*-
# @Time     : 2019/1/29 16:46
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

from q2.MyMethod import *
import numpy as np
from rnn_utils import *
import random

if __name__ == '__main__':

    # load data
    data = open('dinos.txt', 'r').read()
    data = data.lower()
    chars = list(set(data))
    print(sorted(chars))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

    # mapping
    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    print(ix_to_char)

    # test clip avoid exloarding gradient
    np.random.seed(3)
    dWax = np.random.randn(5, 3) * 10
    dWaa = np.random.randn(5, 5) * 10
    dWya = np.random.randn(2, 5) * 10
    db = np.random.randn(5, 1) * 10
    dby = np.random.randn(2, 1) * 10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    gradients =clip(gradients,10)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
