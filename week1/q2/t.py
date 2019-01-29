# -*- coding: utf-8 -*-
# @Time     : 2019/1/29 20:31
# @Author   : HuangYin
# @FileName : t.py
# @Software : PyCharm
import numpy as np

if __name__ == '__main__':
    x = np.array([[1, 2, 3, 5, 6, 7, 8, 9], [111, 2, 3, 5, 6, 7, 8, 9]])
    for i in x:
        # np.clip(i, -3, 9, out=i)
        i[1] = 3333
        print("for zhong ",i)
    print(x)
