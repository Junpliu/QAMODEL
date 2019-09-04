#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils.evaluate_utils import metric, total_metric


MODEL_PATH = "./model/"
RESULT_PATH = "./result/"
model_version = "20190809"


def pair_metric(model_name, threshold):
    pred_file = MODEL_PATH + model_name + "/" + model_version + "/best_eval_loss/output_test"
    metric(pred_file, threshold)


def query_metric(model_name, threshold):
    pred_file = MODEL_PATH + model_name + "/" + model_version + "/best_eval_loss/output_test"
    result_file = RESULT_PATH + model_name + "/" + model_version + "/result.txt"
    total_metric(pred_file, result_file, threshold)


def pair_query_metric(model_name, threshold):
    pair_metric(model_name, threshold)
    query_metric(model_name, threshold)


pair_query_metric("FastText", 0.8)
pair_query_metric("SCWLSTM", 0.8)
pair_query_metric("TextCNN", 0.8)

pair_query_metric("APCNN", 0.8)
pair_query_metric("APCNNT", 0.8)

pair_query_metric("APLSTM", 0.8)
pair_query_metric("APLSTMT", 0.8)

pair_query_metric("ARCII", 0.8)
pair_query_metric("ARCIIT", 0.8)

# pair_query_metric("BiMPM", 0.8)
pair_query_metric("ESIM", 0.8)
