# /usr/bin/python
# -*- coding:utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions

Update: aitingliu, 20190731
"""
import os
from collections import Counter
import logging
import argparse

import pandas as pd
import xgboost as xgb

import data_helper
from common import common_function

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--train_data_file", type=str, default='../../data/20190726/raw/train.txt', help="Train set.")
parser.add_argument("--val_data_file", type=str, default='../../data/20190726/raw/dev.txt', help="Train set.")
parser.add_argument("--test_data_file", type=str, default='../../data/20190726/raw/test.txt', help="Train set.")
parser.add_argument("--pred_data_file", type=str, default='tmp/results_xgb.csv', help="Prediction file.")
parser.add_argument("--word_counts_file", type=str, default='tmp/word_counts', help="Word counts file.")
parser.add_argument("--feature_map_file", type=str, default='tmp/feature_map', help="Feature map file.")
parser.add_argument("--train_scwlstm_pred_file", type=str, default="../../src/model/SCWLSTM/best_eval_loss/output_train")
parser.add_argument("--val_scwlstm_pred_file", type=str, default="../../src/model/SCWLSTM/best_eval_loss/output_dev")
parser.add_argument("--test_scwlstm_pred_file", type=str, default="../../src/model/SCWLSTM/best_eval_loss/output_test")
parser.add_argument("--use_scwlstm", type=bool, default=False)
parser.add_argument("--model_path", type=str, default="model/model")
parser.add_argument("--embedding_dim", type=int, default=200)
parser.add_argument("--embedding_file", type=str, default='../../wdic/word2vec.dict')
parser.add_argument("--stopword_file", type=str, default='../../wdic/stopwords.txt')
args = parser.parse_args()

common_function.print_args(args)

common_function.makedir(args.pred_data_file)
common_function.makedir(args.word_counts_file)
common_function.makedir(args.stopword_file)
common_function.makedir(args.model_path)

#################################################################

logger.info("Starting to read Embedding file...")
word2vec = common_function.load_word2vec(args.embedding_file, filter_num=args.embedding_dim)
logger.info("Finish reading Embedding file !")
logger.info('Found %d word vectors of word2vec' % len(word2vec))

stop_words = common_function.load_file_2_dict(args.stopword_file, colum=1)
logger.info("Finish reading stopword file !")
logger.info('Stopword is : ' + "|".join(list(stop_words.keys())))

############# reading data  #################################################
logger.info("Starting to read training samples...")
train_texts_1, train_texts_2, labels = data_helper.read_data(args.train_data_file)
val_texts_1, val_texts_2, val_labels = data_helper.read_data(args.val_data_file)
test_texts_1, test_texts_2, test_labels = data_helper.read_data(args.test_data_file)
logger.info("Finish reading training samples !")

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
r = open(args.word_counts_file, 'w', encoding="utf-8")
for _word, _count in counts.items():
    r.write("%s\t%d\n" % (_word, _count))
r.close()

################ basic features  ########################################
train_cp = train_orig.copy()
val_cp = val_orig.copy()
test_cp = test_orig.copy()
x_train_basic = data_helper.get_basic_feat(train_cp, args.embedding_dim, stop_words, word2vec)
x_valid_basic = data_helper.get_basic_feat(val_cp, args.embedding_dim, stop_words, word2vec)
x_test_basic = data_helper.get_basic_feat(test_cp, args.embedding_dim, stop_words, word2vec)

############## sentence word char features #################################
weights = {word: data_helper.get_weight(count) for word, count in counts.items()}
x_train_more = data_helper.build_features(train_orig, stop_words, weights)
x_valid_more = data_helper.build_features(val_orig, stop_words, weights)
x_test_more = data_helper.build_features(test_orig, stop_words, weights)

################### SCWLSTM model simscore ####################
if args.use_scwlstm:
    x_train_sim = data_helper.model_simscore(args.train_scwlstm_pred_file, train_cp)
    x_valid_sim = data_helper.model_simscore(args.val_scwlstm_pred_file, val_cp)
    x_test_sim = data_helper.model_simscore(args.test_scwlstm_pred_file, test_cp)

############## combine all features ########################################
    x_train = pd.concat((x_train_basic, x_train_more, x_train_sim), axis=1)
    x_valid = pd.concat((x_valid_basic, x_valid_more, x_valid_sim), axis=1)
    x_test = pd.concat((x_test_basic, x_test_more, x_test_sim), axis=1)
else:
    x_train = pd.concat((x_train_basic, x_train_more), axis=1)
    x_valid = pd.concat((x_valid_basic, x_valid_more), axis=1)
    x_test = pd.concat((x_test_basic, x_test_more), axis=1)

x_train.drop(['question1', 'question2'], axis=1, inplace=True)
x_valid.drop(['question1', 'question2'], axis=1, inplace=True)
x_test.drop(['question1', 'question2'], axis=1, inplace=True)

# print(x_train.columns)

features = [x for x in x_train.columns]
data_helper.ceate_feature_map(args.feature_map_file, features)

x_train.columns = [str(i) for i in range(x_train.shape[1])]
x_valid.columns = [str(i) for i in range(x_valid.shape[1])]
x_test.columns = [str(i) for i in range(x_test.shape[1])]

############## train models ################################################
params = {
    "booster": "gbtree",  # [default= gbtree ]

    "eta": 0.3,  # [default=0.3]
    "gamma": 0,  # [default=0]
    "max_depth": 6,  # [default=6]
    "min_child_weight": 1,  # [default=1]
    "max_delta_step": 0,  # [default=0]
    "subsample": 1,  # [default=1]
    "colsample_bytree": 1,  # [default=1]
    "colsample_bylevel": 1,  # [default=1]
    "lambda": 1,  # [default=1]
    "alpha": 0,  # [default=0]
    "scale_pos_weight": 1,  # [default=1]
    "objective": "binary:logistic",
    "eval_metric": ['error', 'logloss']
}

d_train = xgb.DMatrix(x_train, label=labels)
d_valid = xgb.DMatrix(x_valid, label=val_labels)
# logger.info(d_train.feature_names)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50)
bst.save_model(args.model_path)
bst.dump_model(args.model_path + '.dump')

## make the submission
p_test = bst.predict(xgb.DMatrix(x_test))
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(args.pred_data_file,
              header=False, index=False, encoding='utf-8', sep="\t",
              columns=['user_query', 'candidate_query', 'label', 'score'])

## make the submission for best
p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=bst.best_ntree_limit)
df_sub = pd.DataFrame(
    {'user_query': test_texts_1, 'candidate_query': test_texts_2, 'label': test_labels, 'score': p_test.ravel()})
df_sub.to_csv(args.pred_data_file + "_best",
              header=False, index=False, encoding='utf-8', sep="\t",
              columns=['user_query', 'candidate_query', 'label', 'score'])

logger.info("best_iteration: {}".format(bst.best_iteration))
logger.info("ntree_limit=bst.best_ntree_limit: {}".format(bst.best_ntree_limit))
logger.info("best_score: {}".format(bst.best_score))
