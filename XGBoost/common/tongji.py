#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, re

stop_words = {}

model_word = {}
# vec word
for line in open("vec_word.txt"):
    try:
        word, num = line.strip().split("\t")
    except:
        continue

    if word in stop_words:
        continue
    model_word.setdefault(word, int(num))

total_word_num = 0
total_count_num = 0
has_word_num = 0
has_count_num = 0
# model word
for line in open("model_word.txt"):
    try:
        word, num = line.strip().split("\t")
    except:
        continue

    if word in stop_words:
        continue

    if word in model_word:
        has_word_num += 1
        has_count_num += int(num)

    total_word_num += 1
    total_count_num += int(num)

print("has_word_num:{}".format(has_word_num))
print("has_count_num:{}".format(has_count_num))
print("total_word_num:{}".format(total_word_num))
print("total_count_num:{}".format(total_count_num))
print("word_ratio:{}".format(float(has_word_num) / float(total_word_num)))
print("count_ratio:{}".format(float(has_count_num) / float(total_count_num)))
