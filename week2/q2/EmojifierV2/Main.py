# -*- coding: utf-8 -*-
# @Time     : 2019/1/31 23:26
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

from week2.q2.EmojifierV2.Method import *

import numpy as np

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

np.random.seed(1)
from emo_utils import *

if __name__ == '__main__':
    # load data
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

    # test sentences_to_indices
    X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
    print("X1 =", X1)
    print("X1_indices =", X1_indices)

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

    # load data
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/test.csv')
    maxLen = len(max(X_train, key=len).split())

    model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train_indices = sentences_to_indices(X_train, word_to_index, max_len=maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)

    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)

    pred = model.predict(X_test_indices)
    for i in range(len(X_test)):
        # x = X_test_indices
        num = np.argmax(pred[i])
        if num != Y_test[i]:
            print('Expected emoji:' + label_to_emoji(Y_test[i]) + ' prediction: ' + X_test[i] + label_to_emoji(
                num).strip())

    x_test = np.array([' good good good'])
    X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
