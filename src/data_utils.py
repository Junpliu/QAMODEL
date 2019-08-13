#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import logging

from utils import data_helper
from utils import vocab_utils
from utils import misc_utils

logger = logging.getLogger(__name__)

RAW_DATA_PATH = "../data/20190809/raw/"
TRIPLET_DATA_PATH = "../data/20190809/triplet/"
FOR_TRAIN_PATH = "../data/20190809/for_train/"
FOR_BERT_PATH = "../data/20190809/for_bert/"

misc_utils.makedir(TRIPLET_DATA_PATH)
misc_utils.makedir(FOR_TRAIN_PATH)
misc_utils.makedir(FOR_BERT_PATH)

######################  split data  ############################
#
# raw_data_file = RAW_DATA_PATH + "merge_20190809_check_seg.txt"
# train_data_file = RAW_DATA_PATH + "train.txt"
# dev_data_file = RAW_DATA_PATH + "dev.txt"
# test_data_file = RAW_DATA_PATH + "test.txt"
# # TODO: 测试集固定！！！
# data_helper.split_data(raw_data_file, train_data_file, dev_data_file, test_data_file, dev_num=10000, test_num=0)
#
# ##############  generate triplet data and corresponding tumple data  ##########
#
# raw_data_file = RAW_DATA_PATH + "merge_20190809_check_seg.txt"
# triplet_data_file = TRIPLET_DATA_PATH + "triplet.txt"
# for_train_data_file = FOR_TRAIN_PATH + "all.txt"
# data_helper.data_to_triplet(raw_data_file, triplet_data_file, for_train_data_file)
#
# train_data_file = RAW_DATA_PATH + "train.txt"
# train_triplet_data_file = TRIPLET_DATA_PATH + "train.txt"
# for_train_train = FOR_TRAIN_PATH + "train.txt"
# data_helper.data_to_triplet(train_data_file, train_triplet_data_file, for_train_train)
#
# dev_data_file = RAW_DATA_PATH + "dev.txt"
# dev_triplet_data_file = TRIPLET_DATA_PATH + "dev.txt"
# for_train_dev = FOR_TRAIN_PATH + "dev.txt"
# data_helper.data_to_triplet(dev_data_file, dev_triplet_data_file, for_train_dev)
#
# test_data_file = RAW_DATA_PATH + "test.txt"
# test_triplet_data_file = TRIPLET_DATA_PATH + "test.txt"
# for_train_test = FOR_TRAIN_PATH + "test.txt"
# data_helper.data_to_triplet(test_data_file, test_triplet_data_file, for_train_test)

"""
## 20190709
../data/20190709/raw/merge_20190709_seg.txt
# qq pair total 140135
# qq pair positive 48360
# qq pair negative 91775
# query total 29279
# query has positive question 20100
# query dont have positive question 9179
# (query,question+,question-) 81608
# (query,question+, 1) 36415
# (query,question-, 0) 48203
# (query,question,label) 84618


../data/20190709/raw/train.txt
# qq pair total 112109
# qq pair positive 38393
# qq pair negative 73716
# query total 23424
# query has positive question 15990
# query dont have positive question 7434
# (query,question+,question-) 65140
# (query,question+, 1) 29021
# (query,question-, 0) 38431
# (query,question,label) 67452


../data/20190709/raw/dev.txt
# qq pair total 14013
# qq pair positive 4990
# qq pair negative 9023
# query total 2932
# query has positive question 2050
# query dont have positive question 882
# (query,question+,question-) 8168
# (query,question+, 1) 3691
# (query,question-, 0) 4833
# (query,question,label) 8524


../data/20190709/raw/test.txt
# qq pair total 14013
# qq pair positive 4977
# qq pair negative 9036
# query total 2924
# query has positive question 2061
# query dont have positive question 863
# (query,question+,question-) 8297
# (query,question+, 1) 3700
# (query,question-, 0) 4939
# (query,question,label) 8639
"""
"""
## 20190726
../data/20190726/raw/merge_20190726_seg.txt
# qq pair total 196048
# qq pair positive 63923
# qq pair negative 132125
# query total 40941
# query has positive question 27196
# query dont have positive question 13745
# (query,question+,question-) 111549
# (query,question+, 1) 49038
# (query,question-, 0) 66769
# (query,question,label) 115807


../data/20190726/raw/train.txt
# qq pair total 156840
# qq pair positive 51233
# qq pair negative 105607
# query total 32750
# query has positive question 21802
# query dont have positive question 10948
# (query,question+,question-) 89490
# (query,question+, 1) 39352
# (query,question-, 0) 53555
# (query,question,label) 92907


../data/20190726/raw/dev.txt
# qq pair total 19604
# qq pair positive 6397
# qq pair negative 13207
# query total 4102
# query has positive question 2722
# query dont have positive question 1380
# (query,question+,question-) 11045
# (query,question+, 1) 4884
# (query,question-, 0) 6650
# (query,question,label) 11534


../data/20190726/raw/test.txt
# qq pair total 19604
# qq pair positive 6293
# qq pair negative 13311
# query total 4091
# query has positive question 2673
# query dont have positive question 1418
# (query,question+,question-) 11009
# (query,question+, 1) 4802
# (query,question-, 0) 6563
# (query,question,label) 11365
"""
"""
# total	207794
# train	197794
# dev	10000
# test	0
../data/20190809/raw/merge_20190809_check_seg.txt
# qq pair total 207794
# qq pair positive 68510
# qq pair negative 139284
# query total 42601
# query has positive question 28941
# query dont have positive question 13660
# (query,question+,question-) 122229
# (query,question+, 1) 52954
# (query,question-, 0) 72822
# (query,question,label) 125776


../data/20190809/raw/train.txt
# qq pair total 197794
# qq pair positive 65161
# qq pair negative 132633
# query total 40551
# query has positive question 27541
# query dont have positive question 13010
# (query,question+,question-) 116424
# (query,question+, 1) 50419
# (query,question-, 0) 69336
# (query,question,label) 119755


../data/20190809/raw/dev.txt
# qq pair total 10000
# qq pair positive 3349
# qq pair negative 6651
# query total 2051
# query has positive question 1400
# query dont have positive question 651
# (query,question+,question-) 5803
# (query,question+, 1) 2535
# (query,question-, 0) 3484
# (query,question,label) 6019


../data/20190809/raw/test.txt
# qq pair total 10000
# qq pair positive 3229
# qq pair negative 6771
# query total 2057
# query has positive question 1366
# query dont have positive question 691
# (query,question+,question-) 5813
# (query,question+, 1) 2532
# (query,question-, 0) 3429
# (query,question,label) 5961
"""
######################  create vocabulary  ############################
# create word/char level vocabulary based on texts in train data.

# train_data_file = TRIPLET_DATA_PATH + "train.txt"
# word_index_file = TRIPLET_DATA_PATH + "word.txt"
# char_index_file = TRIPLET_DATA_PATH + "char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)
#
# train_data_file = RAW_DATA_PATH + "train.txt"
# word_index_file = RAW_DATA_PATH + "word.txt"
# char_index_file = RAW_DATA_PATH + "char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)
#
# ######################  test batch_iterator()  ############################
# data_helper.test_batch_iter("../data/20190809/raw/toy.txt", "../data/20190809/raw/word.txt", "../data/20190809/raw/char.txt")
# data_helper.test_triplet_batch_iter("../data/20190809/triplet/toy.txt", "../data/20190809/triplet/word.txt", "../data/20190809/triplet/char.txt")


# TODO: generate data for BERT
"""
query\tquestion\tlabel
id\tquery\tquestion\tlabel
"""
train_df = pd.read_csv(RAW_DATA_PATH + "train.txt", sep="\t", header=None, names=["q1", "q2", "label"])
train_df["q1_char"] = train_df.q1.map(lambda x: " ".join(vocab_utils.text_to_char_list(x)))
train_df["q2_char"] = train_df.q2.map(lambda x: " ".join(vocab_utils.text_to_char_list(x)))
train_df.to_csv(FOR_BERT_PATH + "train.csv", sep="\t", index=True, header=False, columns=["q1_char", "q2_char", "label"])

dev_df = pd.read_csv(RAW_DATA_PATH + "dev.txt", sep="\t", header=None, names=["q1", "q2", "label"])
dev_df["q1_char"] = dev_df.q1.map(lambda x: " ".join(vocab_utils.text_to_char_list(x)))
dev_df["q2_char"] = dev_df.q2.map(lambda x: " ".join(vocab_utils.text_to_char_list(x)))
dev_df.to_csv(FOR_BERT_PATH + "dev.csv", sep="\t", index=True, header=False, columns=["q1_char", "q2_char", "label"])

test_df = pd.read_csv(RAW_DATA_PATH + "test.txt", sep="\t", header=None, names=["q1", "q2", "label"])
test_df["q1_char"] = test_df.q1.map(lambda x: " ".join(vocab_utils.text_to_char_list(x)))
test_df["q2_char"] = test_df.q2.map(lambda x: " ".join(vocab_utils.text_to_char_list(x)))
test_df.to_csv(FOR_BERT_PATH + "test.csv", sep="\t", index=True, header=False, columns=["q1_char", "q2_char", "label"])

