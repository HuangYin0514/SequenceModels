# -*- coding: utf-8 -*-
# @Time     : 2019/2/3 13:55
# @Author   : HuangYin
# @FileName : main.py
# @Software : PyCharm

from week3.q1.method import *
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Xoh.shape:", Xoh.shape)
    print("Yoh.shape:", Yoh.shape)

    # look pre-process the result
    index = 0
    print("Source date:", dataset[index][0])
    print("Target date:", dataset[index][1])
    print()
    print("Source after preprocessing (indices):", X[index])
    print("Target after preprocessing (indices):", Y[index])
    print()
    print("Source after preprocessing (one-hot):", Xoh[index])
    print("Target after preprocessing (one-hot):", Yoh[index])

    # test model
    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    model.summary()

    out = model.compile(
        optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
        metrics=['accuracy'],
        loss='categorical_crossentropy'
    )

    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))

    outputs = list(Yoh.swapaxes(0, 1))

    model.fit([Xoh, s0, c0], outputs, epochs=0, batch_size=100)

    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    EXAMPLES = ['3 May 1979']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
        source = np.expand_dims(source, axis=0)
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output))
