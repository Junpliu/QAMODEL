#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Utilities for data process and load.

Author: aitingliu
Date: 2019-07-08
"""
import random
import logging
import collections

import numpy as np
import sklearn

from . import vocab_utils
# import vocab_utils

logger = logging.getLogger(__name__)


########################################  read big file   ########################################
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


########################################  split data   ########################################

def split_data(raw_file, train_file, dev_file, test_file, dev_ratio=0.1, test_ratio=0.1):
    """split data into train/dev/test"""
    d = {}
    for line in read_file(raw_file):
        query, question, label = line.strip().split("\t")
        if query not in d:
            d[query] = []
        else:
            d[query].append("{}\t{}".format(question, label))
    # TODO: shuffled data
    test = list(d.items())
    random.shuffle(test)
    d = dict(test)
    lines = []
    for k, v in d.items():
        for i in v:
            line = "{}\t{}\n".format(k, i)
            lines.append(line)
    total = len(lines)
    dev_num = int(total * dev_ratio)
    test_num = int(total * test_ratio)
    train = lines[:-(test_num + dev_num)]
    dev = lines[-(test_num + dev_num):-test_num]
    test = lines[-test_num:]
    print("# train\tset\t%d" % len(train))
    print("# dev\tset\t%d" % len(dev))
    print("# test\tset\t%d" % len(test))
    open(train_file, "w", encoding="utf-8").writelines(train)
    open(dev_file, "w", encoding="utf-8").writelines(dev)
    open(test_file, "w", encoding="utf-8").writelines(test)


# raw_data_file = "../data/qq_simscore/merge_20190508.txt"
# train_data_file = "../data/qq_simscore/train.txt"
# dev_data_file = "../data/qq_simscore/dev.txt"
# test_data_file = "../data/qq_simscore/test.txt"
# split_data(raw_data_file, train_data_file, dev_data_file, test_data_file)


########################################  create vocabulary  ########################################
# create word/char level vocabulary based on texts in train data.

# train_data_file = "../data/qq_simscore/train.txt"
# word_index_file = "../data/qq_simscore/word.txt"
# char_index_file = "../data/qq_simscore/char.txt"
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
        word_len1, word_indexlist1 = vocab_utils.list_pad(word_indexlist1, max_len=w_max_len1,
                                                          pad_id=vocab_utils.PAD_ID)
        word_len2, word_indexlist2 = vocab_utils.list_pad(word_indexlist2, max_len=w_max_len2,
                                                          pad_id=vocab_utils.PAD_ID)

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
        char_len1, char_indexlist1 = vocab_utils.list_pad(char_indexlist1, max_len=c_max_len1,
                                                          pad_id=vocab_utils.PAD_ID)
        char_len2, char_indexlist2 = vocab_utils.list_pad(char_indexlist2, max_len=c_max_len2,
                                                          pad_id=vocab_utils.PAD_ID)

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
        mode: train/eval/infer
    Returns:
        A batch iterator for date set.
    """
    data_size = len(data[-1])
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
            print("[batch %d] " % i)
            print(a)
            i += 1
        except StopIteration:
            break


# test_batch_iter()


########################################  triplet data  ########################################
def data_to_triplet(in_path, out_path):
    """
    [raw]  query\tquestion\tlabel
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|350|保养|灯|归零	0
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|怎么|连|蓝牙	0
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|电瓶|在|哪里	0
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|保养|归零|方法	1
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|空调滤芯|在|哪里	0

    -> [triplet]  query\t正例question\t负例question
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|保养|归零|方法   奔驰gl|350|保养|灯|归零
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|保养|归零|方法   奔驰gl|450|怎么|连|蓝牙
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|保养|归零|方法   奔驰gl|450|电瓶|在|哪里
    07|年|奔驰gl|450|保养|归零|方法	奔驰gl|450|保养|归零|方法   奔驰gl|450|空调滤芯|在|哪里

    """

    f = open(in_path, "r", encoding="utf-8")
    fw = open(out_path, "w", encoding="utf-8")

    c1, c0 = 0, 0  # 统计正负例数量
    d = {}
    for line in f:
        line = line.strip()
        q1, q2, label = line.split("\t")
        if q1 not in d:
            d[q1] = [[], []]  # 对于每个query维护一个二维列表， list[0]仍是一个列表，存储所有负例，list[1]存储所有正例
        if label == "0":
            d[q1][0].append(q2)
            c0 += 1
        elif label == "1":
            d[q1][1].append(q2)
            c1 += 1
    f.close()

    total = len(d)
    count0 = 0
    count1 = 0  # query对应的question中至少有1个正例的数量

    # TODO: shuffled data
    # test = list(d.items())
    # random.shuffle(test)
    # d = dict(test)
    count = 0
    for k, v in d.items():
        if len(v[1]) > 0:
            count1 += 1
            if len(v[0]) > 0:
                for i in range(len(v[1])):
                    for j in range(len(v[0])):
                        tri = "{}\t{}\t{}\n".format(k, v[1][i], v[0][j])
                        count += 1
                        fw.write(tri)
        else:
            count0 += 1
    fw.close()

    print("# qq pair total %d" % (c1 + c0))
    print("# qq pair positive %d" % c1)
    print("# qq pair negative %d" % c0)
    print("# query total %d" % total)
    print("# query has positive question %d" % count1)
    print("# query dont have positive question %d" % count0)
    print("# (query,question+,question-) %d" % count)


# raw_data_file = "../data/qq_simscore/merge_20190508.txt"
# triplet_data_file = "../data/qq_simscore/triplet/triplet.txt"
# data_to_triplet(raw_data_file, triplet_data_file)


def split_triplet_data(raw_file, train_file, dev_file, test_file, infer_file, dev_ratio=0.1, test_ratio=0.1):
    """split data into train/dev/test"""
    d = {}
    for line in read_file(raw_file):
        query, q1, q2 = line.strip().split("\t")
        if query not in d:
            d[query] = []
        else:
            d[query].append("{}\t{}".format(q1, q2))
    # TODO: shuffled data
    test = list(d.items())
    random.shuffle(test)
    d = dict(test)
    lines = []
    for k, v in d.items():
        for i in v:
            line = "{}\t{}\n".format(k, i)
            lines.append(line)
    total = len(lines)
    dev_num = int(total * dev_ratio)
    test_num = int(total * test_ratio)
    train = lines[:-(test_num + dev_num)]
    dev = lines[-(test_num + dev_num):-test_num]
    test = lines[-test_num:]

    infer = set()
    for line in test:
        q, q1, q2 = line.strip().split("\t")
        infer.add("{}\t{}\t{}\n".format(q, q1, "1"))
        infer.add("{}\t{}\t{}\n".format(q, q2, "0"))
    infer = list(infer)

    print("# train\tset\t%d" % len(train))
    print("# dev\tset\t%d" % len(dev))
    print("# test\tset\t%d" % len(test))
    open(train_file, "w", encoding="utf-8").writelines(train)
    open(dev_file, "w", encoding="utf-8").writelines(dev)
    open(test_file, "w", encoding="utf-8").writelines(test)
    open(infer_file, "w", encoding="utf-8").writelines(infer)


# raw_data_file = "../data/qq_simscore/triplet/triplet.txt"
# train_data_file = "../data/qq_simscore/triplet/train.txt"
# dev_data_file = "../data/qq_simscore/triplet/dev.txt"
# test_data_file = "../data/qq_simscore/triplet/test.txt"
# infer_data_file = "../data/qq_simscore/triplet/infer.txt"
# split_triplet_data(raw_data_file, train_data_file, dev_data_file, test_data_file, infer_data_file)


class Data(collections.namedtuple("Data", ("words1", "words_len1", "chars1", "chars_len1",
                                           "words2", "words_len2", "chars2", "chars_len2",
                                           "words3", "words_len3", "chars3", "chars_len3",
                                           "labels"))):
    pass


def load_triplet_data(data_file,
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

    words1, chars1, words_len1, chars_len1 = [], [], [], []
    words2, chars2, words_len2, chars_len2 = [], [], [], []
    words3, chars3, words_len3, chars_len3 = [], [], [], []
    labels = []

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
            label = int(line[2])
            labels.append(label)
        else:
            text1, text2, text3 = line.split(split)

        ## text to list
        wordlist1 = vocab_utils.text_to_word_list(text1, split=text_split)
        wordlist2 = vocab_utils.text_to_word_list(text2, split=text_split)

        ## word list to word index
        word_indexlist1 = vocab_utils.list_to_index(wordlist1, word_index, unk_id=vocab_utils.UNK_ID)
        word_indexlist2 = vocab_utils.list_to_index(wordlist2, word_index, unk_id=vocab_utils.UNK_ID)

        # print(wordlist1)
        # print(wordlist2)

        ## list padding
        word_len1, word_indexlist1 = vocab_utils.list_pad(word_indexlist1, max_len=w_max_len1,
                                                          pad_id=vocab_utils.PAD_ID)
        word_len2, word_indexlist2 = vocab_utils.list_pad(word_indexlist2, max_len=w_max_len2,
                                                          pad_id=vocab_utils.PAD_ID)

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
        char_len1, char_indexlist1 = vocab_utils.list_pad(char_indexlist1, max_len=c_max_len1,
                                                          pad_id=vocab_utils.PAD_ID)
        char_len2, char_indexlist2 = vocab_utils.list_pad(char_indexlist2, max_len=c_max_len2,
                                                          pad_id=vocab_utils.PAD_ID)

        # print(char_len1, char_indexlist1)
        # print(char_len2, char_indexlist2)

        chars1.append(char_indexlist1)
        chars2.append(char_indexlist2)
        chars_len1.append(char_len1)
        chars_len2.append(char_len2)

        # TODO: text3
        if mode != "infer":
            wordlist3 = vocab_utils.text_to_word_list(text3, split=text_split)
            word_indexlist3 = vocab_utils.list_to_index(wordlist3, word_index, unk_id=vocab_utils.UNK_ID)
            word_len3, word_indexlist3 = vocab_utils.list_pad(word_indexlist3, max_len=w_max_len2,
                                                              pad_id=vocab_utils.PAD_ID)
            words3.append(word_indexlist3)
            words_len3.append(word_len3)

            charlist3 = vocab_utils.text_to_char_list(text3, split=text_split)
            char_indexlist3 = vocab_utils.list_to_index(charlist3, char_index, unk_id=vocab_utils.UNK_ID)
            char_len3, char_indexlist3 = vocab_utils.list_pad(char_indexlist3, max_len=c_max_len2,
                                                              pad_id=vocab_utils.PAD_ID)
            chars3.append(char_indexlist3)
            chars_len3.append(char_len3)

        if i % 10000 == 0:
            logger.info("  [%s] line %d." % (data_file, i))
    # print(max(words_len1))
    # print(max(words_len2))
    # print(max(chars_len1))
    # print(max(chars_len2))
    if mode == "infer":
        data = [np.array(x, dtype=np.int32) for x in
                [words1, words2, words_len1, words_len2,
                 chars1, chars2, chars_len1, chars_len2,
                 labels]]
    else:
        data = [np.array(x, dtype=np.int32) for x in
                [words1, words2, words3, words_len1, words_len2, words_len3,
                 chars1, chars2, chars3, chars_len1, chars_len2, chars_len3]]
    return data


def triplet_batch_iterator(data, batch_size, shuffle=True, mode="train"):
    """
    Args:
        data: list of arrays.
        batch_size:  batch size.
        shuffle: shuffle or not (default: True)
        mode: train/eval/infer
    Returns:
        A batch iterator for date set.
    """
    data_size = len(data[-1])
    if shuffle:
        if mode == "infer":
            words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2, labels = data
            data = \
                sklearn.utils.shuffle(
                    words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2, labels)
        else:
            words1, words2, words3, words_len1, words_len2, words_len3, chars1, chars2, chars3, chars_len1, chars_len2, chars_len3 = data
            data = \
                sklearn.utils.shuffle(
                    words1, words2, words3, words_len1, words_len2, words_len3,
                    chars1, chars2, chars3, chars_len1, chars_len2, chars_len3)
    num_batches_per_epoch = int((data_size - 1) / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield [x[start_index:end_index] for x in data]
    return


### create vocabulary
train_data_file = "../data/qq_simscore/triplet/train.txt"
infer_data_file = "../data/qq_simscore/triplet/infer.txt"
word_index_file = "../data/qq_simscore/triplet/word.txt"
char_index_file = "../data/qq_simscore/triplet/char.txt"
# vocab_utils.create_vocab_from_triplet_data(train_data_file, word_index_file, split="|", char_level=False)
# vocab_utils.create_vocab_from_triplet_data(train_data_file, char_index_file, split="|", char_level=True)


# train_data = load_triplet_data(train_data_file, word_index_file, char_index_file, mode="train")
# train_iter = triplet_batch_iterator(train_data, 2, mode="train")
# while True:
#     b = next(train_iter)
#     print(b)

# infer_data = load_triplet_data(infer_data_file, word_index_file, char_index_file, mode="infer")
# print(len(infer_data))
# infer_iter = triplet_batch_iterator(infer_data, 2, mode="infer")
# while True:
#     b = next(infer_iter)
#     print(b[-1])
