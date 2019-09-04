#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility to handle vocabulary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

UNK = "<unk>"
PAD = "<pad>"
UNK_ID = 0
PAD_ID = 1

Specials = ['<unk>', '<pad>']


####################  tokenizer from mulanhou  #####################
# def cut_sentence(sentence):
#     """
#      Cut the sentence into the format we want:
#     - continous letters and symbols like back-slash and parenthese
#     - single Chineses character  - other symbols
#     """
#
#     regex = []
#     regex += [r'[0-9a-zA-Z\\+\-<>.]+']  # English and number part for type name.
#     regex += [r'[\u4e00-\ufaff]']       # Chinese characters part.
#     regex += [r'[^\s]']                 # Exclude the space.
#     regex = '|'.join(regex)
#     _RE = re.compile(regex)
#     segs = _RE.findall(sentence.strip())
#     return " ".join(segs)

#################################  create word/char level vocabulary  #######################
def text_to_word_list(text, split="|"):
    """Converts a text to a sequence of words (or tokens)."""
    seq = text.split(split)
    return [i for i in seq if i]


def text_to_char_list(text, split="|"):
    """Converts a text to a sequence of chars."""
    seq = text.split(split)
    seq = [split.join(list(word)) for word in seq if word]
    text = split.join(seq)
    seq = text.split(split)
    return [i for i in seq if i]


def create_vocab(in_path, out_path, max_size=None, min_freq=1, split="|", char_level=False):
    """
    Create vocabulary if not exists.
    :param in_path:
    :param out_path:
    :param max_size: the maximum number of words to keep, based on word frequency.
                    Only the most common `num_words` words will be kept.
    :param min_freq: ignore words which appear less than min_freq.
    :param char_level: if True, every character will be treated as a token.
    :param split: str. Separator for word splitting.
    :return:
    """
    if not os.path.exists(out_path):
        logger.info("Creating vocabulary {} from data {}".format(out_path, in_path))
        vocab = collections.Counter()
        with open(in_path, mode='r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                text1, text2 = line.split("\t")[0], line.split("\t")[1]
                if char_level:
                    vocab.update(text_to_char_list(text1, split=split))
                    vocab.update(text_to_char_list(text2, split=split))
                else:
                    vocab.update(text_to_word_list(text1, split=split))
                    vocab.update(text_to_word_list(text2, split=split))

            sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
            sorted_vocab.sort(key=lambda x: x[1], reverse=True)
            # TODO: special token "<unk>" and "<pad>" is reserved.
            id2word = list(Specials)
            for word, freq in sorted_vocab:
                if min_freq and freq < min_freq or len(id2word) == max_size:
                    break
                id2word.append(word)
            with open(out_path, mode='w', encoding="utf-8") as fw:
                for word in id2word:
                    fw.write(str(word) + '\n')
                    # fw.write(str(word) + '\t' + str(freq) + '\n')

        logger.info("  min frequency %d, vocabulary size %d" % (min_freq, len(id2word)))
        logger.info("  Done create vocab.")
    else:
        logger.info("Vocab file %s already exists." % out_path)


def create_vocab_from_triplet_data(in_path, out_path, max_size=None, min_freq=1, split="|", char_level=False):
    """
    Create vocabulary if not exists.
    :param in_path:
    :param out_path:
    :param max_size: the maximum number of words to keep, based on word frequency.
                    Only the most common `num_words` words will be kept.
    :param min_freq: ignore words which appear less than min_freq.
    :param char_level: if True, every character will be treated as a token.
    :param split: str. Separator for word splitting.
    :return:
    """
    if not os.path.exists(out_path):
        logger.info("Creating vocabulary {} from data {}".format(out_path, in_path))
        vocab = collections.Counter()
        with open(in_path, mode='r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                text1, text2, text3 = line.strip().split("\t")
                if char_level:
                    vocab.update(text_to_char_list(text1, split=split))
                    vocab.update(text_to_char_list(text2, split=split))
                    vocab.update(text_to_char_list(text3, split=split))
                else:
                    vocab.update(text_to_word_list(text1, split=split))
                    vocab.update(text_to_word_list(text2, split=split))
                    vocab.update(text_to_word_list(text3, split=split))

            sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
            sorted_vocab.sort(key=lambda x: x[1], reverse=True)
            # TODO: special token "<unk>" and "<pad>" is reserved.
            id2word = list(Specials)
            for word, freq in sorted_vocab:
                if min_freq and freq < min_freq or len(id2word) == max_size:
                    break
                id2word.append(word)
            with open(out_path, mode='w', encoding="utf-8") as fw:
                for word in id2word:
                    fw.write(str(word) + '\n')
                    # fw.write(str(word) + '\t' + str(freq) + '\n')

        logger.info("  min frequency %d, vocabulary size %d" % (min_freq, len(id2word)))
        logger.info("  Done create vocab.")
    else:
        logger.info("Vocab file %s already exists." % out_path)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a list."""
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


#######################  text to index  ##########################
def load_word_index(vocab_file):
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_size = 0
        for line in f:
            vocab_size += 1
            vocab.append(line.strip())

    word_index = dict(
        list(zip(vocab, list(range(0, len(vocab))))))
    # index_word = dict((c, w) for w, c in word_index.items())
    return word_index


def list_to_index(sequence, word_index, unk_id):
    res = []
    for w in sequence:
        if w in word_index:
            res.append(word_index[w])
        else:
            res.append(unk_id)
    return res


def list_pad(sequence, max_len, pad_id=PAD_ID):
    """Pads word/char sequence to max_len"""
    if len(sequence) >= max_len:
        length = max_len
        sequence = sequence[:max_len]
    else:
        length = len(sequence)
        sequence.extend([pad_id] * (max_len - length))
    return length, sequence


###############################  embedding  ########################################
def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.

     Note: the embed_file should be a Glove/word2vec formatted txt file. Assuming
     Here is an example assuming embed_size=5:

     the -0.071549 0.093459 0.023738 -0.090339 0.056123
     to 0.57346 0.5417 -0.23477 -0.3624 0.4037
     and 0.20327 0.47348 0.050877 0.002103 0.060547

     For word2vec format, the first line will be: <num_words> <emb_size>.

     Args:
       embed_file: file path to the embedding file.
     Returns:
       a dictionary that maps word to vector, and the size of embedding dimensions.
     """
    emb_dict = dict()
    emb_size = None

    is_first_line = True
    with open(embed_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.rstrip().split(" ")
            if is_first_line:
                is_first_line = False
                if len(tokens) == 2:  # header line
                    emb_size = int(tokens[1])
                    continue
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                if emb_size != len(vec):
                    logger.info(
                        "Ignoring %s since embedding size is inconsistent." % word)
                    del emb_dict[word]
            else:
                emb_size = len(vec)
    return emb_dict, emb_size

#################  test  ###################
# print(list_pad([1,2,3], 4))
