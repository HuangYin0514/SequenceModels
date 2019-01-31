# -*- coding: utf-8 -*-
# @Time     : 2019/1/31 21:29
# @Author   : HuangYin
# @FileName : MyMethod.py
# @Software : PyCharm
from w2v_utils import *
import numpy as np


def cosine_similarity(u, v):
    """
       Cosine similarity reflects the degree of similariy between u and v

       Arguments:
           u -- a word vector of shape (n,)
           v -- a word vector of shape (n,)

       Returns:
           cosine_similarity -- the cosine similarity between u and v defined by the formula above.
       """
    distance = 0.0
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    norm_v = np.sqrt(np.sum(np.power(v, 2)))
    cosine_similarity = dot / np.dot(norm_u, norm_v)
    return cosine_similarity


def compute_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
        Performs the word analogy task as explained above: a is to b as c is to ____.

        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors.

        Returns:
        best_word --  bthe word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
        """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None

    for w in words:
        if w in [word_a, word_b, word_c]:
            continue
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[w] - e_c))
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


def neutralize(word, g, word_to_vec_map):
    """
      Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
      This function ensures that gender neutral words are zero in the gender subspace.

      Arguments:
          word -- string indicating the word to debias
          g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
          word_to_vec_map -- dictionary mapping words to their corresponding vectors.

      Returns:
          e_debiased -- neutralized word vector representation of the input "word"
      """
    e = word_to_vec_map[word]
    e_biascomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(g))) * g
    e_debiased = e - e_biascomponent
    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    通过遵循上图中所描述的均衡方法来消除性别偏差。

    参数：
        pair -- 要消除性别偏差的词组，比如 ("actress", "actor")
        bias_axis -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_1 -- 第一个词的词向量
        e_2 -- 第二个词的词向量
    """
    # 第1步：获取词向量
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # 第2步：计算w1与w2的均值
    mu = (e_w1 + e_w2) / 2.0

    # 第3步：计算mu在偏置轴与正交轴上的投影
    mu_B = np.divide(np.dot(mu, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    mu_orth = mu - mu_B

    # 第4步：使用公式7、8计算e_w1B 与 e_w2B
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis

    # 第5步：根据公式9、10调整e_w1B 与 e_w2B的偏置部分
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w1B - mu_B,
                                                                                          np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w2B - mu_B,
                                                                                          np.abs(e_w2 - mu_orth - mu_B))

    # 第6步： 使e1和e2等于它们修正后的投影之和，从而消除偏差
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2

