# -*- coding: utf-8 -*-
# @Time     : 2019/1/31 21:29
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

from week2.q1.MyMethod import *
from w2v_utils import *
import numpy as np

if __name__ == '__main__':
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    """
    #test cosine_similarity
    father = word_to_vec_map["father"]
    mother = word_to_vec_map["mother"]
    ball = word_to_vec_map["ball"]
    crocodile = word_to_vec_map["crocodile"]
    france = word_to_vec_map["france"]
    italy = word_to_vec_map["italy"]
    paris = word_to_vec_map["paris"]
    rome = word_to_vec_map["rome"]
    print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
    print("cosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
    print("cosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))
    """
    # test compute_analogy
    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'),
                     ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print('{} -> {} ::{} ->{}'.format(*triad, compute_analogy(*triad, word_to_vec_map)))


    # woman - man
    g = word_to_vec_map['woman'] - word_to_vec_map['man']
    print(g)

    print('List of names and their similarities with constructed vector:')
    # girls and boys name
    name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
    for w in name_list:
        print(w, cosine_similarity(word_to_vec_map[w], g))

    print('Other words and their similarities:')
    word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior', 'doctor', 'tree', 'receptionist',
                 'technology', 'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
    for w in word_list:
        print(w, cosine_similarity(word_to_vec_map[w], g))


    # test neutralize
    e = "receptionist"
    print("cosine similarity between " + e + " and g, before neutralizing: ",
          cosine_similarity(word_to_vec_map["receptionist"], g))
    e_debiased = neutralize("receptionist", g, word_to_vec_map)
    print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

    # test equalize
    print("cosine similarities before equalizing:")
    print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
    print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
    print()
    e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
    print("cosine similarities after equalizing:")
    print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
    print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))