#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2018-09-06
Author: aiting
"""
import os
import logging

import tensorflow as tf

from utils import misc_utils as utils
from utils import model_helper
from utils import data_helper

logger = logging.getLogger(__name__)


def run_infer(config, loaded_infer_model, infer_sess, pred_file):
    logger.info("  inference to output %s." % pred_file)

    infer_data = data_helper.load_data(config.infer_file, config.word_vocab_file, config.char_vocab_file,
                                       w_max_len1=config.max_word_len1,
                                       w_max_len2=config.max_word_len2,
                                       c_max_len1=config.max_char_len1,
                                       c_max_len2=config.max_char_len2,
                                       text_split="|", split="\t",
                                       mode="infer")
    infer_iterator = data_helper.batch_iterator(infer_data, batch_size=config.infer_batch_size, shuffle=False, mode="infer")

    pred_labels = []
    lines = open(config.infer_file, "r", encoding="utf-8").readlines()

    with open(pred_file, mode="w", encoding="utf-8") as pred_f:
        pred_f.write("")

        while True:
            try:
                batch = next(infer_iterator)
                b_word_ids1, b_word_ids2, b_word_len1, b_word_len2, b_char_ids1, b_char_ids2, b_char_len1, b_char_len2 = batch
                pred = loaded_infer_model.infer(infer_sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2, b_char_ids1, b_char_ids2, b_char_len1, b_char_len2)
                pred_labels.extend(pred)
            except StopIteration:
                logger.info("  Done inference.")
                break

        for line, p in zip(lines, pred_labels):
            res = line.strip() + "\t" + str(p) + "\n"
            pred_f.write(res)


def inference(config, model_creator):
    output_file = "output_" + os.path.split(config.infer_file)[-1].split(".")[0]
    # Inference output directory
    pred_file = os.path.join(config.model_dir, output_file)
    utils.makedir(pred_file)

    # Inference
    model_dir = config.best_eval_loss_dir

    # Create model
    # model_creator = my_model.MyModel
    infer_model = model_helper.create_model(model_creator, config, mode="infer")

    # TensorFlow model
    sess_config = utils.get_config_proto()
    infer_sess = tf.Session(config=sess_config, graph=infer_model.graph)

    with infer_model.graph.as_default():
        loaded_infer_model, _ = model_helper.create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer")

    run_infer(config, loaded_infer_model, infer_sess, pred_file)
