#!/usr/bin/env python
# -*- coding:utf-8 -*-
from utils import data_helper
from utils import vocab_utils
import logging

logger = logging.getLogger(__name__)


######################  split data  ############################

# raw_data_file = "../data/20190709/raw/merge_20190709_seg.txt"
# train_data_file = "../data/20190709/raw/train.txt"
# dev_data_file = "../data/20190709/raw/dev.txt"
# test_data_file = "../data/20190709/raw/test.txt"
# data_helper.split_data(raw_data_file, train_data_file, dev_data_file, test_data_file, dev_ratio=0.1, test_ratio=0.1)

# raw_data_file = "../data/20190726/raw/merge_20190726_seg.txt"
# train_data_file = "../data/20190726/raw/train.txt"
# dev_data_file = "../data/20190726/raw/dev.txt"
# test_data_file = "../data/20190726/raw/test.txt"
# data_helper.split_data(raw_data_file, train_data_file, dev_data_file, test_data_file, dev_ratio=0.1, test_ratio=0.1)

##############  generate triplet data and corresponding tumple data  ##########

# raw_data_file = "../data/20190709/raw/merge_20190709_seg.txt"
# triplet_data_file = "../data/20190709/triplet/triplet.txt"
# for_train_data_file = "../data/20190709/for_train/all.txt"
# data_helper.data_to_triplet(raw_data_file, triplet_data_file, for_train_data_file)
#
# train_data_file = "../data/20190709/raw/train.txt"
# train_triplet_data_file = "../data/20190709/triplet/train.txt"
# for_train_train = "../data/20190709/for_train/train.txt"
# data_helper.data_to_triplet(train_data_file, train_triplet_data_file, for_train_train)
#
# dev_data_file = "../data/20190709/raw/dev.txt"
# dev_triplet_data_file = "../data/20190709/triplet/dev.txt"
# for_train_dev = "../data/20190709/for_train/dev.txt"
# data_helper.data_to_triplet(dev_data_file, dev_triplet_data_file, for_train_dev)
#
# test_data_file = "../data/20190709/raw/test.txt"
# test_triplet_data_file = "../data/20190709/triplet/test.txt"
# for_train_test = "../data/20190709/for_train/test.txt"
# data_helper.data_to_triplet(test_data_file, test_triplet_data_file, for_train_test)

# raw_data_file = "../data/20190726/raw/merge_20190726_seg.txt"
# triplet_data_file = "../data/20190726/triplet/triplet.txt"
# for_train_data_file = "../data/20190726/for_train/all.txt"
# data_helper.data_to_triplet(raw_data_file, triplet_data_file, for_train_data_file)
#
# train_data_file = "../data/20190726/raw/train.txt"
# train_triplet_data_file = "../data/20190726/triplet/train.txt"
# for_train_train = "../data/20190726/for_train/train.txt"
# data_helper.data_to_triplet(train_data_file, train_triplet_data_file, for_train_train)
#
# dev_data_file = "../data/20190726/raw/dev.txt"
# dev_triplet_data_file = "../data/20190726/triplet/dev.txt"
# for_train_dev = "../data/20190726/for_train/dev.txt"
# data_helper.data_to_triplet(dev_data_file, dev_triplet_data_file, for_train_dev)
#
# test_data_file = "../data/20190726/raw/test.txt"
# test_triplet_data_file = "../data/20190726/triplet/test.txt"
# for_train_test = "../data/20190726/for_train/test.txt"
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

######################  create vocabulary  ############################
# create word/char level vocabulary based on texts in train data.

# train_data_file = "../data/20190709/triplet/train.txt"
# word_index_file = "../data/20190709/triplet/word.txt"
# char_index_file = "../data/20190709/triplet/char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)
#
# train_data_file = "../data/20190709/raw/train.txt"
# word_index_file = "../data/20190709/raw/word.txt"
# char_index_file = "../data/20190709/raw/char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)
#
# train_data_file = "../data/20190726/triplet/train.txt"
# word_index_file = "../data/20190726/triplet/word.txt"
# char_index_file = "../data/20190726/triplet/char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)
#
# train_data_file = "../data/20190726/raw/train.txt"
# word_index_file = "../data/20190726/raw/word.txt"
# char_index_file = "../data/20190726/raw/char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)


######################  test batch_iterator()  ############################
data_helper.test_batch_iter("../data/20190709/raw/toy.txt", "../data/20190709/raw/word.txt", "../data/20190709/raw/char.txt")
data_helper.test_triplet_batch_iter("../data/20190709/triplet/toy.txt", "../data/20190709/triplet/word.txt", "../data/20190709/triplet/char.txt")
#
# data_helper.test_batch_iter("../data/20190726/raw/toy.txt", "../data/20190726/raw/word.txt", "../data/20190726/raw/char.txt")
# data_helper.test_triplet_batch_iter("../data/20190726/triplet/toy.txt", "../data/20190726/triplet/word.txt", "../data/20190726/triplet/char.txt")
