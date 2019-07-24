#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys, os, re
import json
import sys
import pandas as pd
from sklearn import metrics


def metric(input_path, thre):
    # model_name = sys.argv[1]
    # thre = float(sys.argv[2])
    f = pd.read_csv(input_path, sep='\t', encoding="utf-8", names=["s1", "s2", "label", "score"])

    f.loc[f.score > thre, 'pred'] = 1
    f.loc[f.score <= thre, 'pred'] = 0
    # print(f.describe())
    print(input_path, thre)
    print("acc ：%.4f" % metrics.accuracy_score(f.label, f.pred))
    print("pre ：%.4f" % metrics.precision_score(f.label, f.pred))
    print("rec ：%.4f" % metrics.recall_score(f.label, f.pred))
    print("f1  ：%.4f" % metrics.f1_score(f.label, f.pred))
    print("auc ：%.4f" % metrics.roc_auc_score(f.label, f.score))
    print("\n")


def total_metric(input_path, output_path, thr):
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    # thr = float(sys.argv[3])
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

        predict_label = 1 if float(predict) > thr else 0  # best answer的结果 是否过阈值
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
    print("P, TP, FN:", P, TP, FN)
    print("N, TN, FP:", N, TN, FP)

    print("Query级别结果指标")
    print("THR={} P={} N={} TP={} FN={} FP={} TN={}".format(thr, P, N, TP, FN, FP, TN))
    print("ACC : %.4f" % ACC)
    print("PRE : %.4f" % PRE)
    print("REC : %.4f" % REC)
    print("F1  : %.4f" % F1)
    print("\n")



# metric("./model/APCNN/output_test", 0.1)
# metric("./model/APCNN/output_test", 0.2)
# metric("./model/APCNN/output_test", 0.3)
# metric("./model/APCNN/output_test", 0.4)
# metric("./model/APCNN/output_test", 0.5)
# metric("./model/APCNN/output_test", 0.6)
# metric("./model/APCNN/output_test", 0.7)
# metric("./model/APCNN/output_test", 0.8)
# metric("./model/APCNN/output_test", 0.9)

# metric("./model/APLSTM/output_test", 0.1)
# metric("./model/APLSTM/output_test", 0.2)
# metric("./model/APLSTM/output_test", 0.3)
# metric("./model/APLSTM/output_test", 0.4)
# metric("./model/APLSTM/output_test", 0.5)
# metric("./model/APLSTM/output_test", 0.6)
# metric("./model/APLSTM/output_test", 0.7)
# metric("./model/APLSTM/output_test", 0.8)
# metric("./model/APLSTM/output_test", 0.9)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.5)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.6)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.7)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.8)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.9)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.95)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.98)
# total_metric("./model/APLSTM/output_test", "./result/qq_simscore/APLSTM/result.txt", 0.99)


# metric("./model/FastText/output_test", 0.1)
# metric("./model/FastText/output_test", 0.2)
# metric("./model/FastText/output_test", 0.3)
# metric("./model/FastText/output_test", 0.4)
# metric("./model/FastText/output_test", 0.5)
# metric("./model/FastText/output_test", 0.6)
# metric("./model/FastText/output_test", 0.7)
# metric("./model/FastText/output_test", 0.8)
# metric("./model/FastText/output_test", 0.9)
# total_metric("./model/FastText/output_test", "./result/qq_simscore/FastText/result.txt", 0.5)
# total_metric("./model/FastText/output_test", "./result/qq_simscore/FastText/result.txt", 0.6)
# total_metric("./model/FastText/output_test", "./result/qq_simscore/FastText/result.txt", 0.7)
# total_metric("./model/FastText/output_test", "./result/qq_simscore/FastText/result.txt", 0.8)
# total_metric("./model/FastText/output_test", "./result/qq_simscore/FastText/result.txt", 0.9)
# total_metric("./model/FastText/output_test", "./result/qq_simscore/FastText/result.txt", 0.95)

# metric("./model/TextCNN/output_test", 0.1)
# metric("./model/TextCNN/output_test", 0.2)
# metric("./model/TextCNN/output_test", 0.3)
# metric("./model/TextCNN/output_test", 0.4)
# metric("./model/TextCNN/output_test", 0.5)
# metric("./model/TextCNN/output_test", 0.6)
# metric("./model/TextCNN/output_test", 0.7)
# metric("./model/TextCNN/output_test", 0.8)
# metric("./model/TextCNN/output_test", 0.9)
# total_metric("./model/TextCNN/output_test", "./result/qq_simscore/TextCNN/result.txt", 0.5)
# total_metric("./model/TextCNN/output_test", "./result/qq_simscore/TextCNN/result.txt", 0.6)
# total_metric("./model/TextCNN/output_test", "./result/qq_simscore/TextCNN/result.txt", 0.7)
# total_metric("./model/TextCNN/output_test", "./result/qq_simscore/TextCNN/result.txt", 0.8)
# total_metric("./model/TextCNN/output_test", "./result/qq_simscore/TextCNN/result.txt", 0.9)
# total_metric("./model/TextCNN/output_test", "./result/qq_simscore/TextCNN/result.txt", 0.95)

# metric("./model/SCWLSTM/output_test", 0.1)
# metric("./model/SCWLSTM/output_test", 0.2)
# metric("./model/SCWLSTM/output_test", 0.3)
# metric("./model/SCWLSTM/output_test", 0.4)
# metric("./model/SCWLSTM/output_test", 0.5)
# metric("./model/SCWLSTM/output_test", 0.6)
# metric("./model/SCWLSTM/output_test", 0.7)
# metric("./model/SCWLSTM/output_test", 0.8)
# metric("./model/SCWLSTM/output_test", 0.9)
# metric("./model/SCWLSTM/output_test", 0.95)
# total_metric("./model/SCWLSTM/output_test", "./result/qq_simscore/SCWLSTM/result.txt", 0.5)
# total_metric("./model/SCWLSTM/output_test", "./result/qq_simscore/SCWLSTM/result.txt", 0.6)
# total_metric("./model/SCWLSTM/output_test", "./result/qq_simscore/SCWLSTM/result.txt", 0.7)
# total_metric("./model/SCWLSTM/output_test", "./result/qq_simscore/SCWLSTM/result.txt", 0.8)
# total_metric("./model/SCWLSTM/output_test", "./result/qq_simscore/SCWLSTM/result.txt", 0.9)
# total_metric("./model/SCWLSTM/output_test", "./result/qq_simscore/SCWLSTM/result.txt", 0.95)

# metric("./model/ARCII/output_test", 0.1)
# metric("./model/ARCII/output_test", 0.2)
# metric("./model/ARCII/output_test", 0.3)
# metric("./model/ARCII/output_test", 0.4)
# metric("./model/ARCII/output_test", 0.5)
# metric("./model/ARCII/output_test", 0.6)
# metric("./model/ARCII/output_test", 0.7)
# metric("./model/ARCII/output_test", 0.8)
# metric("./model/ARCII/output_test", 0.9)
# metric("./model/ARCII/output_test", 0.95)
# metric("./model/ARCII/output_test", 0.99)
# total_metric("./model/ARCII/output_test", "./result/qq_simscore/ARCII/result.txt", 0.5)
# total_metric("./model/ARCII/output_test", "./result/qq_simscore/ARCII/result.txt", 0.6)
# total_metric("./model/ARCII/output_test", "./result/qq_simscore/ARCII/result.txt", 0.7)
# total_metric("./model/ARCII/output_test", "./result/qq_simscore/ARCII/result.txt", 0.8)
# total_metric("./model/ARCII/output_test", "./result/qq_simscore/ARCII/result.txt", 0.9)
# total_metric("./model/ARCII/output_test", "./result/qq_simscore/ARCII/result.txt", 0.95)

# metric("./model/BiMPM/output_test", 0.1)
# metric("./model/BiMPM/output_test", 0.2)
# metric("./model/BiMPM/output_test", 0.3)
# metric("./model/BiMPM/output_test", 0.4)
# metric("./model/BiMPM/output_test", 0.5)
# metric("./model/BiMPM/output_test", 0.6)
# metric("./model/BiMPM/output_test", 0.7)
# metric("./model/BiMPM/output_test", 0.8)
# metric("./model/BiMPM/output_test", 0.9)
# metric("./model/BiMPM/output_test", 0.95)
# metric("./model/BiMPM/output_test", 0.99)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.5)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.6)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.7)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.8)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.9)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.95)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.98)
# total_metric("./model/BiMPM/output_test", "./result/qq_simscore/BiMPM/result.txt", 0.99)
