# /usr/bin/python
# -*- coding:utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions
"""
import os

import pandas as pd
import xgboost as xgb

import data_helper
from common import common_function

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

TEST_DATA_FILE = '../../data/partition/test.txt'
PRE_DATA_FILE = '../../data/partition/results_xgb.csv_tmp_'
WORD_COUNTS = '../../data/word_counts'

MODEL_PATH = 'model/model'

EMBEDDING_DIM = 200
EMBEDDING_FILE = '../../wdic/word2vec.dict'
STOPWORD_FILE = '../../wdic/stopwords.txt'

#################################################################
print("Starting to read Embedding file...")
word2vec = common_function.load_word2vec(EMBEDDING_FILE, filter_num=EMBEDDING_DIM)
print("Finish reading Embedding file !")
print('Found %d word vectors of word2vec' % len(word2vec))

stop_words = common_function.load_file_2_dict(STOPWORD_FILE, colum=1)
print("Finish reading stopword file !")
print('Stopword is : ' + "|".join(list(stop_words.keys())))

############# reading data  #################################################
print("Starting to read training samples...")
test_texts_1, test_texts_2, test_labels = data_helper.read_data(TEST_DATA_FILE)
test_orig = pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})
print("Finish reading training samples !")

############### read words counts #########################################
counts = common_function.load_file_2_dict(WORD_COUNTS)
weights = {word: data_helper.get_weight(int(count)) for word, count in counts.items()}

################ make features  ########################################
test_cp = test_orig.copy()
x_test_basic = data_helper.get_basic_feat(test_cp, EMBEDDING_DIM, stop_words, word2vec)
x_test_more = data_helper.build_features(test_orig, stop_words, weights)

############## combine all features ########################################
x_test = pd.concat((x_test_basic, x_test_more), axis=1)
x_test.columns = [str(i) for i in range(x_test.shape[1])]
print("x_test shape : ", x_test.shape)

################ save DMatrix binary data to make loading faster #########
xgb.DMatrix(x_test).save_binary('test.buffer')

############## predict models ################################################
bst = xgb.Booster({'nthread': 1})  # init model
bst.load_model(MODEL_PATH)  # load data

best_ntree_limit = 647
p_test = bst.predict(xgb.DMatrix(x_test))
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + str(best_ntree_limit),
              index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

best_ntree_limit = 596
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + str(best_ntree_limit),
              index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

best_ntree_limit = 597
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + str(best_ntree_limit),
              index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

best_ntree_limit = 598
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + str(best_ntree_limit),
              index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

best_ntree_limit = 599
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + str(best_ntree_limit),
              index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

p_test = bst.predict(xgb.DMatrix(x_test))
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(PRE_DATA_FILE + "nolimit",
              index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')
