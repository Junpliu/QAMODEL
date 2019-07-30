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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

TRAIN_DATA_FILE = 'data/partition/train.txt'
VAL_DATA_FILE = 'data/partition/dev.txt'
TEST_DATA_FILE = 'data/partition/test.txt'
PRE_DATA_FILE = 'data/partition/results_xgb.csv'
WORD_COUNTS = 'data/word_counts'
FEATURE_MAP = 'data/feature_map'

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
train_texts_1, train_texts_2, labels = read_data(TRAIN_DATA_FILE)
val_texts_1, val_texts_2, val_labels = read_data(VAL_DATA_FILE)
test_texts_1, test_texts_2, test_labels = read_data(TEST_DATA_FILE)
print("Finish reading training samples !")

train_orig = pd.DataFrame({"question1": train_texts_1, "question2": train_texts_2})
val_orig = pd.DataFrame({"question1": val_texts_1, "question2": val_texts_2})
test_orig = pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})

############### save words counts #########################################
total_words = []
ques = pd.concat([train_orig, val_orig], axis=0).reset_index(drop='index')
for i in range(ques.shape[0]):
    total_words += ques.question1[i].split('|')
    total_words += ques.question2[i].split('|')

## save word freq to total_counts
counts = Counter(total_words)
r = open(WORD_COUNTS, 'w')
for _word, _count in counts.items():
    r.write("%s\t%d\n" % (_word, _count))
r.close()

################ basic features  ########################################
train_cp = train_orig.copy()
val_cp = val_orig.copy()
test_cp = test_orig.copy()
x_train_basic = get_basic_feat(train_cp, EMBEDDING_DIM, stop_words, word2vec)
x_valid_basic = get_basic_feat(val_cp, EMBEDDING_DIM, stop_words, word2vec)
x_test_basic = get_basic_feat(test_cp, EMBEDDING_DIM, stop_words, word2vec)

############## sentence word char features #################################
weights = {word: get_weight(count) for word, count in counts.items()}
x_train_more = build_features(train_orig, stop_words, weights)
x_valid_more = build_features(val_orig, stop_words, weights)
x_test_more = build_features(test_orig, stop_words, weights)

############## combine all features ########################################
x_train = pd.concat((x_train_basic, x_train_more), axis=1)
x_valid = pd.concat((x_valid_basic, x_valid_more), axis=1)
x_test = pd.concat((x_test_basic, x_test_more), axis=1)

features = [x for x in x_train.columns]
ceate_feature_map(FEATURE_MAP, features)

x_train.columns = [str(i) for i in range(x_train.shape[1])]
x_valid.columns = [str(i) for i in range(x_valid.shape[1])]
x_test.columns = [str(i) for i in range(x_test.shape[1])]
# print (x_train.columns)
# print (x_train.columns.values)

############## train models ################################################
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['error', 'logloss']
# params['eval_metric'] = ['auc', 'ams@0']
params['eta'] = 0.08
params['max_depth'] = 6

# params['gpu_id'] = 2
# params['max_bin'] = 16
# params['tree_method'] = 'gpu_hist'

d_train = xgb.DMatrix(x_train, label=labels)
d_valid = xgb.DMatrix(x_valid, label=val_labels)
# print (d_train.feature_names)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50)
bst.save_model(MODEL_PATH)
bst.dump_model(MODEL_PATH + '.dump')

## make the submission
p_test = bst.predict(xgb.DMatrix(x_test))
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE, index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

## make the submission for best
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=bst.best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + "_best", index=False, columns=['user_query', 'candidate_query', 'label', 'score'],
              encoding='utf-8')

print("best_iteration:", bst.best_iteration)
print("ntree_limit=bst.best_ntree_limit:", bst.best_ntree_limit)
print("best_score", bst.best_score)
