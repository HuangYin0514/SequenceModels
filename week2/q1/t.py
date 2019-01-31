# -*- coding: utf-8 -*-
# @Time     : 2019/1/31 21:44
# @Author   : HuangYin
# @FileName : t.py
# @Software : PyCharm
import numpy as np

if __name__ == '__main__':
    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'),
                     ('small', 'smaller', 'large')]
    for i in triads_to_try:
        print("i ====",i)
        c = [*i]

        print("*i ====",*i)
