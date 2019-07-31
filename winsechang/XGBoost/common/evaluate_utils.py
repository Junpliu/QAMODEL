#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def metric(input_path, threshold):
    """
    QQ pair级别指标.
    [data format]: query\tquestion\tlabel\tsimscore
        艾滋病|皮肤|初期|症状	艾滋病|的|初期|都|有|什么|症状	1	0.88182044
        开户行|行号|怎么|查询	开户行|的|行号|是|什么	0	9.426536e-18
        张家界|周边|旅游|景点	张家界|最|有名|的|景点	0	0.00023530718
        张家界|旅游|景点|天气	张家界|最|有名|的|景点	0	0.035709225
        银行卡|信息|怎么|查询	银行卡|的|信息|是|什么	0	1.3402569e-16
        路由器|是否|影响|网速	路由器|对|网速|的|影响	1	0.99993587
        四川省|理科|高考|人数	四川省|的|高考|用|什么|s	0	1.36783385e-08
        为什么|唯品|会要|运费	结过婚|的|女人|你|会要|吗	0	6.2627487e-07
    :param input_path:
    :param threshold:
    :return:
    """
    f = pd.read_csv(input_path, sep='\t', encoding="utf-8", names=["s1", "s2", "label", "score"])

    f.loc[f.score > threshold, 'pred'] = 1
    f.loc[f.score <= threshold, 'pred'] = 0
    # print(f.describe())
    print(input_path, threshold)
    print("acc ：%.4f" % metrics.accuracy_score(f.label, f.pred))
    print("pre ：%.4f" % metrics.precision_score(f.label, f.pred))
    print("rec ：%.4f" % metrics.recall_score(f.label, f.pred))
    print("f1  ：%.4f" % metrics.f1_score(f.label, f.pred))
    print("auc ：%.4f" % metrics.roc_auc_score(f.label, f.score))
    print("\n")


def total_metric(input_path, output_path, threshold):
    """
    Query级别指标.
    :param input_path:
    :param output_path:
    :param threshold:
    :return:
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    d = {}
    for line in open(input_path, "r", encoding="utf-8"):
        q1, q2, label, predict = line.strip().split("\t")
        if q1 == "user_query":
            continue
        d.setdefault(q1, [])
        d[q1].append((q2, label, predict))

    labels = []
    scores = []
    pred_labels = []

    f = open(output_path, "w", encoding="utf-8")
    P, N, TP, FN, FP, TN = 0, 0, 0, 0, 0, 0
    for q1 in d:
        terms = d[q1]

        sort_l = [(predict, (q2, label, predict)) for q2, label, predict in terms]  # 针对每一个query的所有question预测结果排序
        sort_l.sort(reverse=True)
        terms_str = "\t".join(["-".join(list(term[1])) for term in sort_l])
        # terms_str = str(sort_l)

        best_term = sort_l[0][1]
        q2, label, predict = best_term
        # print(best_term)

        predict_label = 1 if float(predict) > threshold else 0  # best answer的结果 是否过阈值
        # predict_answer = predict_label * int(label) # 判定best answer预测结果是否正确
        has_answer = int(any([int(label) for q2, label, predict in terms]))  # 判定候选question是否有正确答案
        # print(has_answer)
        # predict_answer has_answer  q1 q2_list 作为最终的判定序列
        f.write("\t".join([str(has_answer), label, str(predict_label), predict, q1, terms_str]) + "\n")

        label = int(label)
        if has_answer == 1: P += 1  # 正类 候选包含正确答案
        if has_answer == 0: N += 1  # 负类 候选不包含正确答案
        if label == 1 and predict_label == 1: TP += 1  # 将正类预测为正类数
        if label == 1 and predict_label == 0: FN += 1  # 将正类预测为负类数
        if label == 0 and predict_label == 1: FP += 1  # 将负类预测为正类数
        if label == 0 and predict_label == 0: TN += 1  # 将负类预测为负类数

        labels.append(label)
        scores.append(predict)
        pred_labels.append(predict_label)

    PRE = TP / (TP + FP + 0.00000001)
    REC = TP / (P + 0.00000001)
    ACC = (TP + TN) / (P + N + 0.00000001)
    F1 = 2 * PRE * REC / (PRE + REC + 0.00000001)

    print("Query级别结果指标")
    print("THRESHOLD={} P={} N={} TP={} FN={} FP={} TN={}".format(threshold, P, N, TP, FN, FP, TN))
    print("ACC : %.4f" % ACC)
    print("PRE : %.4f" % PRE)
    print("REC : %.4f" % REC)
    print("F1  : %.4f" % F1)
    print("\n")


def print_fig(in_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df = pd.read_csv(in_path, sep="\t", encoding="utf-8", names=["q1", "q2", "label", "sim"])
    label = np.array(df.label)
    pred = np.array(df.sim)
    x = np.array(range(len(label)))
    x_ = np.array([i + 0.4 for i in x])
    fig = plt.figure()
    plt.bar(x, label, width=0.4)
    plt.bar(x_, pred, width=0.4)
    print(label, pred)
    fig_name = "fenbu.png"
    plt.savefig(os.path.join(output_path, fig_name))
    plt.show()

