#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-07-31
Author: aitingliu

docker run -p 8501:8501 --mount type=bind,source=/Users/liuaiting/Desktop/QAMODEL/src/model/SCWLSTM/tfmodel,target=/models/tfmodel -e MODEL_NAME=tfmodel -t tensorflow/serving &
"""
import requests
import json
import numpy as np
from utils import data_helper, vocab_utils

# The server URL specifies the endpoint of your server running the NLU
# model with the name "nlu_model" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/tfmodel:predict'


def request(predict_request):
    """
    Send an http request to the server, get a response.
    :param predict_request: JSON
    :return: JSON
    """
    # Send few actual requests and report average latency.
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    # print(response.elapsed.total_seconds())
    outputs = response.json()
    return json.dumps(outputs)


word_index_file = "../data/20190726/raw/word.txt"
char_index_file = "../data/20190726/raw/char.txt"


def tf_model(query, question):
    line = "{}\t{}".format(query, question)
    print(line)
    word_index = vocab_utils.load_word_index(word_index_file)
    char_index = vocab_utils.load_word_index(char_index_file)
    train_data = data_helper.process_line(line,
                                          word_index, char_index,
                                          w_max_len1=10,
                                          w_max_len2=10,
                                          c_max_len1=20,
                                          c_max_len2=20,
                                          text_split="|", split="\t", mode="infer")
    print(train_data)
    res = [[x] for x in train_data if x is not None]

    (word_ids1, word_ids2, word_len1, word_len2,
     char_ids1, char_ids2, char_len1, char_len2) = res
    print(res)

    request_data = json.dumps(
        {
            "inputs":
                {
                    "word_ids1": word_ids1,
                    "word_ids2": word_ids2,
                    "word_len1": word_len1,
                    "word_len2": word_len2,
                    "char_ids1": char_ids1,
                    "char_ids2": char_ids2,
                    "char_len1": char_len1,
                    "char_len2": char_len2
                }
        })
    response = request(request_data)
    # res = json.loads(response)
    # print(res["outputs"])
    return response


tf_model("你是谁", "你说啥")
