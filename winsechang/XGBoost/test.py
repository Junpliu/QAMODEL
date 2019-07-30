# /usr/bin/python
# -*- coding:utf-8 -*-
'''
Author: changjingdong
Date: 20190614
Desc: data processer for xgboost
'''
import sys
import codecs
from collections import defaultdict
from collections import Counter
import pandas as pd
import numpy as np
import functools
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import os

sys.path.insert(0, '../common')
from common_function import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

STOPWORD_FILE = '../../wdic/stopwords.txt'
WORD_COUNTS = 'data/word_counts'


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)


counts = load_file_2_dict(WORD_COUNTS)
print("XXXXXXX : " + str(counts["5"]))

weights = {word: get_weight(int(count)) for word, count in counts.items()}

stop_words = load_file_2_dict(STOPWORD_FILE, colum=1)
print("Finish reading stopword file !")
print('Stopword is : ' + "|".join(list(stop_words.keys())))


def remove_stopwords(s, stops):
    return list_filter(s, lambda x: x not in stops)


def tfidf_word_match_share_stops(q1, q2, weights={}, stops={}):
    q1words = remove_stopwords(set(q1.split('|')), stops)
    q2words = remove_stopwords(set(q2.split('|')), stops)
    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    print("q1words : " + "|".join(q1words))
    print("q2words : " + "|".join(q2words))

    shared_weights = [weights.get(w, 0) for w in q1words if w in q2words] + [weights.get(w, 0) for w in q2words if
                                                                             w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    tmp1 = ["%s-%f" % (w, weights.get(w, 0)) for w in q1words if w in q2words] + ["%s-%f" % (w, weights.get(w, 0)) for w
                                                                                  in q2words if w in q1words]

    tmp2 = ["%s-%f" % (w, weights.get(w, 0)) for w in q1words] \
           + ["%s-%f" % (w, weights.get(w, 0)) for w in q2words]

    tmp3 = ["%s-%s" % (w, counts.get(w, 0)) for w in q1words if w in q2words]

    print(tmp1)
    print(tmp2)
    print(tmp3)

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


q1 = "小米|5|手机|原装|电池|怎么|换"
q2 = "小米|5|手机|能|换|电池|吗"

tfidf = tfidf_word_match_share_stops(q1, q2, weights, stop_words)

print(tfidf)
