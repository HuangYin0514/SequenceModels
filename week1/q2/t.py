# -*- coding: utf-8 -*-
# @Time     : 2019/1/29 20:31
# @Author   : HuangYin
# @FileName : t.py
# @Software : PyCharm
import numpy as np
from q2.MyMethod import *

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

    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    print(examples)
    # X = [None] + [char_to_ix[ch] for ch in examples[1]]
    # Y = X[1:] + [char_to_ix["\n"]]
    # print(X)
    # print(Y)
