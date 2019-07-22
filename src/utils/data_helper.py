#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Utilities for data process and load.

Author: aitingliu
Date: 2019-07-08
"""
import random
import logging

import numpy as np
import sklearn

from . import vocab_utils
# import vocab_utils

logger = logging.getLogger(__name__)


# raw_data_file = "../data/qq_simscore/merge_20190508.txt"
# train_data_file = "../data/qq_simscore/train.txt"
# dev_data_file = "../data/qq_simscore/dev.txt"
# test_data_file = "../data/qq_simscore/test.txt"
# word_index_file = "../data/qq_simscore/word.txt"
# char_index_file = "../data/qq_simscore/char.txt"

########################################  read big file   ########################################
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


########################################  split data   ########################################

def split_data(raw_file, train_file, dev_file, test_file):
    """split data into train/dev/test"""
    lines = open(raw_file, "r", encoding="utf-8").readlines()
    train = lines[:-20000]
    dev = lines[-20000:-10000]
    test = lines[-10000:]
    open(train_file, "w", encoding="utf-8").writelines(train)
    open(dev_file, "w", encoding="utf-8").writelines(dev)
    open(test_file, "w", encoding="utf-8").writelines(test)

# split_data(raw_data_file, train_data_file, dev_data_file, test_data_file)


########################################  create vocabulary  ########################################
# create word/char level vocabulary based on texts in train data.
# vocab_utils.create_vocab(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab(train_data_file, char_index_file, split="|", char_level=True)


#######################################  load data & batch iterator  ########################################

def load_data(data_file,
              word_index_file,
              char_index_file,
              w_max_len1=40,
              w_max_len2=40,
              c_max_len1=40,
              c_max_len2=40,
              text_split="|",
              split="\t",
              mode="train"):
    """load train/dev/test data"""
    logger.info("# Loading data.")
    labels, words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2 = [], [], [], [], [], [], [], [], []
    word_index = vocab_utils.load_word_index(word_index_file)
    char_index = vocab_utils.load_word_index(char_index_file)

    f = read_file(data_file)
    i = 0
    for line in f:
        i += 1
        line = line.strip()
        if mode == "infer":
            line = line.split(split)
            text1, text2 = line[0], line[1]
        else:
            text1, text2, label = line.split(split)
            labels.append(int(label))
        ## text to list
        wordlist1 = vocab_utils.text_to_word_list(text1, split=text_split)
        wordlist2 = vocab_utils.text_to_word_list(text2, split=text_split)

        ## word list to word index
        word_indexlist1 = vocab_utils.list_to_index(wordlist1, word_index, unk_id=vocab_utils.UNK_ID)
        word_indexlist2 = vocab_utils.list_to_index(wordlist2, word_index, unk_id=vocab_utils.UNK_ID)

        # print(wordlist1)
        # print(wordlist2)

        ## list padding
        word_len1, word_indexlist1 = vocab_utils.list_pad(word_indexlist1, max_len=w_max_len1, pad_id=vocab_utils.PAD_ID)
        word_len2, word_indexlist2 = vocab_utils.list_pad(word_indexlist2, max_len=w_max_len2, pad_id=vocab_utils.PAD_ID)

        # print(word_len1, word_indexlist1)
        # print(word_len2, word_indexlist2)

        words1.append(word_indexlist1)
        words2.append(word_indexlist2)
        words_len1.append(word_len1)
        words_len2.append(word_len2)

        ## text to list
        charlist1 = vocab_utils.text_to_char_list(text1, split=text_split)
        charlist2 = vocab_utils.text_to_char_list(text2, split=text_split)

        # print(charlist1)
        # print(charlist2)

        ## char list to char index
        char_indexlist1 = vocab_utils.list_to_index(charlist1, char_index, unk_id=vocab_utils.UNK_ID)
        char_indexlist2 = vocab_utils.list_to_index(charlist2, char_index, unk_id=vocab_utils.UNK_ID)

        ## list padding
        char_len1, char_indexlist1 = vocab_utils.list_pad(char_indexlist1, max_len=c_max_len1, pad_id=vocab_utils.PAD_ID)
        char_len2, char_indexlist2 = vocab_utils.list_pad(char_indexlist2, max_len=c_max_len2, pad_id=vocab_utils.PAD_ID)

        # print(char_len1, char_indexlist1)
        # print(char_len2, char_indexlist2)

        chars1.append(char_indexlist1)
        chars2.append(char_indexlist2)
        chars_len1.append(char_len1)
        chars_len2.append(char_len2)

        if i % 10000 == 0:
            logger.info("  [%s] line %d." % (data_file, i))
    # print(max(words_len1))
    # print(max(words_len2))
    # print(max(chars_len1))
    # print(max(chars_len2))
    if mode == "infer":
        data = [np.array(x, dtype=np.int32) for x in
                [words1, words2, words_len1, words_len2,
                 chars1, chars2, chars_len1, chars_len2]]
    else:
        data = [np.array(x, dtype=np.int32) for x in
                [words1, words2, words_len1, words_len2,
                 chars1, chars2, chars_len1, chars_len2,
                 labels]]
    # for d in data:
    #     print(d)
    return data


def batch_iterator(data, batch_size, shuffle=True, mode="train"):
    """
    Args:
        data: list of arrays.
        batch_size:  batch size.
        shuffle: shuffle or not (default: True)
    Returns:
        A batch iterator for date set.
    """
    data_size = len(data[-1])
    start = 0
    if shuffle:
        if mode == "infer":
            words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2 = data
            data = \
                sklearn.utils.shuffle(
                    words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2)
        else:
            words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2, labels = data
            data = \
                sklearn.utils.shuffle(
                    words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2, labels)
    num_batches_per_epoch = int((data_size - 1) / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield [x[start_index:end_index] for x in data]
    return


def test_batch_iter():
    lines = ["360|系统|怎么|更换|默认|桌面	努比亚|怎么|设置|360|桌面|为|默认	0\n",
             "电动车|驾驶证	2018|电动车|要|驾驶证|吗	0\n",
             "二战|时期|哪个|国家|牺牲|的|人|最多	二战|时期|有没有	0\n"]
    open("../data/qq_simscore/toy.txt", "w", encoding="utf-8").writelines(lines)

    d = load_data("../data/qq_simscore/toy.txt", "../data/qq_simscore/word.txt", "../data/qq_simscore/char.txt")
    it = batch_iterator(d, 2)
    i = 0
    while True:
        try:
            a = next(it)
            print("[batch %d] "%i)
            print(a)
            i += 1
        except StopIteration:
            break

# test_batch_iter()
# data = load_data(train_data_file)