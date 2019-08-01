#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, re
import json
from common_function import *
from functools import reduce

"""
修正后的query级别指标计算
"""

P, N, TP, FN, FP, TN = 0, 0, 0, 0, 0, 0
for line in open(sys.argv[1]):
    has_answer, label, predict_label, predict, q1, terms_str = line.strip().split("\t")
    has_answer = int(has_answer)
    label = int(label)
    predict_label = int(predict_label)

    label = int(label)
    if has_answer == 1: P += 1  # 正类 候选包含正确答案
    if has_answer == 0: N += 1  # 负类 候选不包含正确答案
    if label == 1 and predict_label == 1: TP += 1  # 将正类预测为正类数
    if label == 1 and predict_label == 0: FN += 1  # 将正类预测为负类数
    if label == 0 and predict_label == 1: FP += 1  # 将负类预测为正类数
    if label == 0 and predict_label == 0: TN += 1  # 将负类预测为负类数

PRE = TP / (TP + FP)
REC = TP / P
ACC = (TP + TN) / (P + N)
F1 = 2 / (1 / PRE + 1 / REC)

print("Query级别结果指标[修正后]")
print("P={} N={} TP={} FN={} FP={} TN={}".format(P, N, TP, FN, FP, TN))
print("ACC : %.4f" % ACC)
print("PRE : %.4f" % PRE)
print("REC : %.4f" % REC)
print("F1  : %.4f" % F1)
