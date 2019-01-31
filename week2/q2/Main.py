# -*- coding: utf-8 -*-
# @Time     : 2019/1/31 23:26
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

from week2.q2.Method import *

import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # laod data
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/test.csv')

    # compute the max len of X_train element
    maxLen = len(max(X_train, key=len).split())

    # look at the X_train of element
    index = 1
    print(X_train[index], label_to_emoji(Y_train[index]))

    # convert to one hot
    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)
    index = 50
    print(Y_train[index], "is converted into one hot", Y_oh_train[index])
