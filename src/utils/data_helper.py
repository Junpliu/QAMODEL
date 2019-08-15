#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Utilities for data process and load.

Author: aitingliu
Date: 2019-07-08
"""
import random
import logging
import os

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

def split_data(raw_file, train_file, dev_file, test_file, dev_num=10000, test_num=10000):
    """split data into train/dev/test"""
    # TODO: shuffled data
    d = {}
    for line in read_file(raw_file):
        query, question, label = line.strip().split("\t")
        if query not in d:
            d[query] = []
        d[query].append("{}\t{}".format(question, label))
    test = list(d.items())
    random.shuffle(test)
    d = dict(test)
    lines = []
    for k, v in d.items():
        for i in v:
            line = "{}\t{}\n".format(k, i)
            lines.append(line)

    total = len(lines)
    dev_num = int(dev_num)
    test_num = int(test_num)
    train = lines[:total-(test_num + dev_num)]
    dev = lines[total-(test_num + dev_num):total-test_num]
    test = lines[total-test_num:]
    print("# total\t%d" % total)
    print("# train\t%d" % len(train))
    print("# dev\t%d" % len(dev))
    print("# test\t%d" % len(test))
    open(train_file, "w", encoding="utf-8").writelines(train)
    open(dev_file, "w", encoding="utf-8").writelines(dev)
    # open(test_file, "w", encoding="utf-8").writelines(test)


#######################################  load data & batch iterator  ########################################
def process_line(line,
                 word_index,
                 char_index,
                 w_max_len1=40,
                 w_max_len2=40,
                 c_max_len1=40,
                 c_max_len2=40,
                 text_split="|",
                 split="\t",
                 mode="train"):
    line = line.strip()
    if mode == "infer":
        line = line.split(split)
        text1, text2 = line[0], line[1]
        label = None
    else:
        text1, text2, label = line.split(split)
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

    res = [word_indexlist1, word_indexlist2, word_len1, word_len2,
           char_indexlist1, char_indexlist2, char_len1, char_len2, label]

    return res


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
        word_indexlist1, word_indexlist2, word_len1, word_len2, char_indexlist1, char_indexlist2, char_len1, char_len2, label = \
            process_line(
                line,
                word_index,
                char_index,
                w_max_len1=w_max_len1,
                w_max_len2=w_max_len2,
                c_max_len1=c_max_len1,
                c_max_len2=c_max_len2,
                text_split=text_split,
                split=split,
                mode=mode)

        labels.append(label)

        words1.append(word_indexlist1)
        words2.append(word_indexlist2)
        words_len1.append(word_len1)
        words_len2.append(word_len2)

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
    if shuffle and mode == "train":
        words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2, labels = data
        data = \
            sklearn.utils.shuffle(
                words1, words2, words_len1, words_len2, chars1, chars2, chars_len1, chars_len2, labels)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield [x[start_index:end_index] for x in data]
    return


def test_batch_iter(toy_file, word_file, char_file):
    lines = ["360|系统|怎么|更换|默认|桌面	努比亚|怎么|设置|360|桌面|为|默认	0\n",
             "电动车|驾驶证	2018|电动车|要|驾驶证|吗	0\n",
             "二战|时期|哪个|国家|牺牲|的|人|最多	二战|时期|有没有	0\n"]
    open(toy_file, "w", encoding="utf-8").writelines(lines)

    d = load_data(toy_file, word_file, char_file)
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


########################################  triplet data  ########################################
def data_to_triplet(in_path, out_path, for_train):
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

    Args:
        in_path: pair数据，[raw]  query\tquestion\tlabel
        out_path: triplet数据， [triplet]  query\t正例question\t负例question
        for_train： 把原始文件中只包含负例或只包含正例的query删掉，保证数据一致性
    #TODO：deprecated for_train
    """
    out_path_dir = os.path.dirname(out_path)
    if not os.path.exists(out_path_dir):
        os.makedirs(out_path_dir)
    # for_train_dir = os.path.dirname(for_train)
    # if not os.path.exists(for_train_dir):
    #     os.makedirs(for_train_dir)

    f = open(in_path, "r", encoding="utf-8")
    fw = open(out_path, "w", encoding="utf-8")
    # f2 = open(for_train, "w", encoding="utf-8")

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
    count_p = 0
    count_n = 0
    for k, v in d.items():
        if len(v[1]) > 0:
            count1 += 1
            if len(v[0]) > 0:
                for i in range(len(v[1])):
                    for j in range(len(v[0])):
                        tri = "{}\t{}\t{}\n".format(k, v[1][i], v[0][j])
                        count += 1
                        fw.write(tri)

                for i in range(len(v[1])):
                    tmp = "{}\t{}\t{}\n".format(k, v[1][i], "1")
                    count_p += 1
                    # f2.write(tmp)
                for j in range(len(v[0])):
                    tmp = "{}\t{}\t{}\n".format(k, v[0][j], "0")
                    count_n += 1
                    # f2.write(tmp)

        else:
            count0 += 1
    fw.close()
    # f2.close()
    print(in_path)
    print("# qq pair total %d" % (c1 + c0))
    print("# qq pair positive %d" % c1)
    print("# qq pair negative %d" % c0)
    print("# query total %d" % total)
    print("# query has positive question %d" % count1)
    print("# query dont have positive question %d" % count0)
    print("# (query,question+,question-) %d" % count)
    print("# (query,question+, 1) %d" % count_p)
    print("# (query,question-, 0) %d" % count_n)
    print("# (query,question,label) %d" % (count_n + count_p))
    print("\n")


# class Data(collections.namedtuple("Data", ("words1", "words_len1", "chars1", "chars_len1",
#                                            "words2", "words_len2", "chars2", "chars_len2",
#                                            "words3", "words_len3", "chars3", "chars_len3",
#                                            "labels"))):
#     pass


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
            text1, text2, label = line.split(split)
            labels.append(int(label))
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
    if shuffle and mode == "train":
        words1, words2, words3, words_len1, words_len2, words_len3, chars1, chars2, chars3, chars_len1, chars_len2, chars_len3 = data
        data = \
            sklearn.utils.shuffle(
                words1, words2, words3, words_len1, words_len2, words_len3,
                chars1, chars2, chars3, chars_len1, chars_len2, chars_len3)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield [x[start_index:end_index] for x in data]
    return


def test_triplet_batch_iter(toy_file, word_file, char_file):
    print("# test triplet_batch_iterator(), mode = train")
    lines = ["称象|有|什么|方法	现代|称象|的|方法	曹|什么|称象\n",
             "称象|有|什么|方法	现代|称象|的|方法	用|称象|造句\n",
             "称象|有|什么|方法	现代|称象|的|方法	称象|小|古文\n"]
    open(toy_file, "w", encoding="utf-8").writelines(lines)

    d = load_triplet_data(toy_file, word_file, char_file, mode="train")
    it = triplet_batch_iterator(d, 2, mode="train")
    i = 0
    while True:
        try:
            a = next(it)
            print("[batch %d] " % i)
            print(a)
            i += 1
        except StopIteration:
            break

    print("# test triplet_batch_iterator(), mode = infer")
    lines = ["广东|汤|河粉|什么|配料	广东|卤鹅|的|做法|及|配料	0\n",
             "广东|汤|河粉|什么|配料	广东|窑鸡|的|做法|及|配料	0\n",
             "广东|汤|河粉|什么|配料	广东|河粉|的|制作|方法	1\n"]
    open(toy_file, "w", encoding="utf-8").writelines(lines)

    d = load_triplet_data(toy_file, word_file, char_file, mode="infer")
    it = triplet_batch_iterator(d, 2, mode="infer")
    i = 0
    while True:
        try:
            a = next(it)
            print("[batch %d] " % i)
            print(a)
            i += 1
        except StopIteration:
            break
