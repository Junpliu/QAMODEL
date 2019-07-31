# /usr/bin/python
# -*- coding:utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions

Update: aitingliu, 20190731
"""
import os
import argparse

import pandas as pd
import xgboost as xgb

import data_helper
from common import common_function

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--test_data_file", type=str, default='../../data/20190726/raw/test.txt', help="Train set.")
parser.add_argument("--pred_data_file", type=str, default='tmp/results_xgb.csv_tmp_', help="Prediction file.")
parser.add_argument("--word_counts_file", type=str, default='tmp/word_counts', help="Word counts file.")
parser.add_argument("--test_scwlstm_pred_file", type=str, default="../../src/model/SCWLSTM/best_eval_loss/output_test")
parser.add_argument("--use_scwlstm", type=bool, default=False)
parser.add_argument("--model_path", type=str, default="model/model")
parser.add_argument("--embedding_dim", type=int, default=200)
parser.add_argument("--embedding_file", type=str, default='../../wdic/word2vec.dict')
parser.add_argument("--stopword_file", type=str, default='../../wdic/stopwords.txt')
args = parser.parse_args()

common_function.makedir(args.pred_data_file)

#################################################################
print("Starting to read Embedding file...")
word2vec = common_function.load_word2vec(args.embedding_file, filter_num=args.embedding_dim)
print("Finish reading Embedding file !")
print('Found %d word vectors of word2vec' % len(word2vec))

stop_words = common_function.load_file_2_dict(args.stopword_file, colum=1)
print("Finish reading stopword file !")
print('Stopword is : ' + "|".join(list(stop_words.keys())))

############# reading data  #################################################
print("Starting to read training samples...")
test_texts_1, test_texts_2, test_labels = data_helper.read_data(args.test_data_file)
test_orig = pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})
print("Finish reading training samples !")

############### read words counts #########################################
counts = common_function.load_file_2_dict(args.word_counts_file)
weights = {word: data_helper.get_weight(int(count)) for word, count in counts.items()}

################ make features  ########################################
test_cp = test_orig.copy()
x_test_basic = data_helper.get_basic_feat(test_cp, args.embedding_dim, stop_words, word2vec)
x_test_more = data_helper.build_features(test_orig, stop_words, weights)
if args.use_scwlstm:
    x_test_sim = data_helper.model_simscore(args.test_scwlstm_pred_file, test_cp)

############## combine all features ########################################
    x_test = pd.concat([x_test_basic, x_test_more, x_test_sim], axis=1)
else:
    x_test = pd.concat((x_test_basic, x_test_more), axis=1)
x_test.columns = [str(i) for i in range(x_test.shape[1])]
print("x_test shape : ", x_test.shape)

################ save DMatrix binary data to make loading faster #########
xgb.DMatrix(x_test).save_binary('test.buffer')

############## predict models ################################################
bst = xgb.Booster()  # init model
bst.load_model(args.model_path)  # load model

best_ntree_limit = 634
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(args.pred_data_file + str(best_ntree_limit),
              header=False, index=False, encoding='utf-8', sep="\t",
              columns=['user_query', 'candidate_query', 'label', 'score'])

best_ntree_limit = 584
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(args.pred_data_file + str(best_ntree_limit),
              header=False, index=False, encoding='utf-8', sep="\t",
              columns=['user_query', 'candidate_query', 'label', 'score'])
