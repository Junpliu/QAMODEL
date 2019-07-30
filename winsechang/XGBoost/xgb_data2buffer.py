# /usr/bin/python
# -*- coding:utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions
"""
import sys
import codecs
from collections import defaultdict
from collections import Counter
import pandas as pd
import numpy as np
import functools
import xgboost as xgb
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import pickle as pk
import os
from data_helper import *

sys.path.insert(0, '../common')
from common_function import *

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
pd.set_option('precision', 14)

TEST_DATA_FILE = 'data/partition/test.txt'
PRE_DATA_FILE = 'data/partition/results_xgb.csv_tmp_'
WORD_COUNTS = 'data/word_counts'

MODEL_PATH = 'model/model'

EMBEDDING_DIM = 200
EMBEDDING_FILE = '../../wdic/word2vec.dict'
STOPWORD_FILE = '../../wdic/stopwords.txt'

#################################################################
print("Starting to read Embedding file...")
word2vec = load_word2vec(EMBEDDING_FILE, filter_num=EMBEDDING_DIM)
print("Finish reading Embedding file !")
print('Found %d word vectors of word2vec' % len(word2vec))

stop_words = load_file_2_dict(STOPWORD_FILE, colum=1)
print("Finish reading stopword file !")
print('Stopword is : ' + "|".join(list(stop_words.keys())))

############# reading data  #################################################
print("Starting to read training samples...")
test_texts_1, test_texts_2, test_labels = read_data(TEST_DATA_FILE)
test_orig = pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})
print("Finish reading training samples !")

############### read words counts #########################################
counts = load_file_2_dict(WORD_COUNTS)
weights = {word: get_weight(int(count)) for word, count in counts.items()}

################ make features  ########################################
test_cp = test_orig.copy()
x_test_basic = get_basic_feat(test_cp, EMBEDDING_DIM, stop_words, word2vec)
x_test_more = build_features(test_orig, stop_words, weights)

############## combine all features ########################################
x_test = pd.concat((x_test_basic, x_test_more), axis=1)
x_test.columns = [str(i) for i in range(x_test.shape[1])]
print("x_test shape : ", x_test.shape)

print(x_test)
x_test.to_csv('tmp/fea.csv', sep=',', header=True, index=True)

################ save DMatrix binary data to make loading faster #########
# xgb.DMatrix(x_test).save_binary('test.buffer_1000')
