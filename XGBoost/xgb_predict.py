#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions

Update: aitingliu, 20190731
"""
import os
import argparse
import logging

import pandas as pd
import xgboost as xgb

import data_helper
from common import common_function

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--test_data_file", type=str, default='/ceph/qbkg2/aitingliu/qq/data/20190726/raw/test.txt')
parser.add_argument("--x_test_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/x_test.csv')
parser.add_argument("--y_test_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/y_test.csv')
parser.add_argument("--dtest_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/dtest.buffer')
parser.add_argument("--pred_data_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/results_xgb.csv_tmp_')
parser.add_argument("--word_counts_file", type=str, default='/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/word_counts')
parser.add_argument("--test_scwlstm_pred_file", type=str, default="/ceph/qbkg2/aitingliu/qq/src/model/SCWLSTM/best_eval_loss/output_test")
parser.add_argument("--use_scwlstm", type=bool, default=True)
parser.add_argument("--model_path", type=str, default="/ceph/qbkg2/aitingliu/qq/XGBoost/model/with_scwlstm/model")
parser.add_argument("--embedding_dim", type=int, default=200)
parser.add_argument("--embedding_file", type=str, default='/ceph/qbkg2/winsechang/MODEL/qq_simscore/wdic/word2vec.dict')
parser.add_argument("--stopword_file", type=str, default='/ceph/qbkg2/winsechang/MODEL/qq_simscore/wdic/stopwords.txt')
parser.add_argument("--ntree_limit", type=int, default=0, help="ntree_limit")
args = parser.parse_args()

common_function.print_args(args)

common_function.makedir(args.pred_data_file)

# TODO: build features the first time to predict
# #################################################################
# print("Starting to read Embedding file...")
# word2vec = common_function.load_word2vec(args.embedding_file, filter_num=args.embedding_dim)
# print("Finish reading Embedding file !")
# print('Found %d word vectors of word2vec' % len(word2vec))
#
# stop_words = common_function.load_file_2_dict(args.stopword_file, colum=1)
# print("Finish reading stopword file !")
# print('Stopword is : ' + "|".join(list(stop_words.keys())))
#
# ############# reading data  #################################################
print("Starting to read testing samples...")
test_texts_1, test_texts_2, test_labels = data_helper.read_data(args.test_data_file)
test_orig = pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})
print("Finish reading testing samples !")
#
# ############### read words counts #########################################
# counts = common_function.load_file_2_dict(args.word_counts_file)
# weights = {word: data_helper.get_weight(int(count)) for word, count in counts.items()}
#
# ################ make features  ########################################
# test_cp = test_orig.copy()
# x_test_basic = data_helper.get_basic_feat(test_cp, args.embedding_dim, stop_words, word2vec)
# x_test_more = data_helper.build_features(test_orig, stop_words, weights)
# if args.use_scwlstm:
#     x_test_sim = data_helper.model_simscore(args.test_scwlstm_pred_file, test_cp)
#
# ############## combine all features ########################################
#     x_test = pd.concat([x_test_basic, x_test_more, x_test_sim], axis=1)
# else:
#     x_test = pd.concat((x_test_basic, x_test_more), axis=1)
#
# x_test.drop(['question1', 'question2'], axis=1, inplace=True)
#
# x_test.columns = [str(i) for i in range(x_test.shape[1])]
# print("x_test shape : ", x_test.shape)
#
# ################ save DMatrix binary data to make loading faster #########
# xgb.DMatrix(x_test).save_binary('test.buffer')
#
# ############## predict models ################################################
# bst = xgb.Booster()  # init model
# bst.load_model(args.model_path)  # load model
#
# p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=args.ntree_limit)
# df_sub = pd.DataFrame(
#     {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
# df_sub.to_csv(args.pred_data_file + str(args.ntree_limit),
#               header=False, index=False, encoding='utf-8', sep="\t",
#               columns=['user_query', 'candidate_query', 'label', 'score'])
#

###################### load a CSV file into DMatrix ######################
# x_test = pd.read_csv(args.x_test_file, header=None, encoding="utf-8", sep="\t")
# y_test = pd.read_csv(args.y_test_file, header=None, encoding="utf-8", sep="\t")
# d_test = xgb.DMatrix(x_test, y_test)

###################### load a XGBoost binary file into DMatrix ######################

d_test = xgb.DMatrix(args.dtest_file)

###################### do predict ###########################
bst = xgb.Booster()  # init model
bst.load_model(args.model_path)  # load model

p_test = bst.predict(d_test, ntree_limit=args.ntree_limit)

df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(args.pred_data_file + str(args.ntree_limit),
              header=False, index=False, encoding='utf-8', sep="\t",
              columns=['user_query', 'candidate_query', 'label', 'score'])
