# -*- coding: utf-8 -*-
# @Time     : 2019/1/30 15:13
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm
from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from q3.MyMethod import *

if __name__ == '__main__':
    # IPython.display.Audio('./data/30s_seq.mp3')

    # load data
    X, Y, n_values, indices_values = load_music_utils()
    print('shape of X:', X.shape)
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', n_values)
    print('Shape of Y:', Y.shape)

    # init parameters
    n_a = 64
    # reshapor = Reshape((1, 78))
    # LSTM_cell = LSTM(n_a, return_state=True)
    # densor = Dense(n_values, activation='softmax')

    # create model
    model, LSTM_cell, densor = djmodel(Tx=30, n_a=64, n_values=78)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # set data
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))

    # training
    model.fit(
        [X, a0, c0],
        list(Y),
        epochs=19
    )

    # generate 50 values
    inference_model = music_inference_model(LSTM_cell, densor)

    x_initializer = np.zeros((1, 1, 78))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    results,indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
    print("np.argmax(results[12]) =", np.argmax(results[12]))
    print("np.argmax(results[17]) =", np.argmax(results[17]))
    print("list(indices[12:18]) =", list(indices[12:18]))


    out_stream = generate_music(inference_model)