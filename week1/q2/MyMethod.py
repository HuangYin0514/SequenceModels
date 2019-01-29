# -*- coding: utf-8 -*-
# @Time     : 2019/1/29 16:46
# @Author   : HuangYin
# @FileName : MyMethod.py
# @Software : PyCharm

import numpy as np
from rnn_utils import *
import random


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def sample(parameters, char_to_ix, seed):
    """
       Sample a sequence of characters according to a sequence of probability distributions output of the RNN

       Arguments:
       parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
       char_to_ix -- python dictionary mapping each character to an index.
       seed -- used for grading purposes. Do not worry about it.

       Returns:
       indices -- a list of length n containing the indices of the sampled characters.
       """
    # retrieve parameters
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # init x and a
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    # init result
    indices = []
    # init index that means flag of end
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        # forward cell
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # to same with blog
        np.random.seed(counter + seed)

        # sample one from y
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        # add idx to result
        indices.append(idx)

        # gener a new x ,which x == y
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # set a
        a_prev = a

        # set seed to same blog
        seed += 1
        counter += 1

    # if counter = 50,which means  the word that does not have flag of end
    # ex.   one word  ==> aklsdjfkjas\n
    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices
