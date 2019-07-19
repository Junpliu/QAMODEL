# -*- coding: utf-8 -*-
"""
Date: 2018-09-06
Author: aiting
"""
from __future__ import division
from __future__ import print_function

import time
import os
import codecs
import logging

import tensorflow as tf

from utils import misc_utils as utils
from utils import model_helper
from utils import data_helper

logger = logging.getLogger(__name__)


def run_eval(config, eval_model, eval_sess, eval_data, model_dir, ckpt_name, save_on_best=True):
    output_file = "output_" + os.path.split(config.dev_file)[-1].split(".")[0]
    pred_file = os.path.join(model_dir, output_file)
    logger.info("  predictions to output %s." % pred_file)

    summary_writer = tf.summary.FileWriter(os.path.join(model_dir, "eval_log"), eval_model.graph)

    eval_iterator = data_helper.batch_iterator(eval_data, batch_size=config.batch_size, shuffle=False)

    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            eval_model.model, model_dir, eval_sess, "eval")

        # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        # running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        # TODO: tf.metrics
        # eval_sess.run(running_vars_initializer)
        eval_sess.run(tf.local_variables_initializer())

    pred_labels = []
    eval_loss = 0.0
    step = 0
    lines = open(config.dev_file, "r", encoding="utf-8").readlines()
    with open(pred_file, mode="w", encoding="utf-8") as pred_f:
        pred_f.write("")
        while True:
            try:
                b_word_ids1, b_word_ids2, b_word_len1, b_word_len2, b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels = next(
                    eval_iterator)
                eval_summary1, pred, step_loss, acc_op, rec_op, pre_op, auc_op = \
                    loaded_eval_model.eval(eval_sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2,
                                           b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels)
                pred_labels.extend(pred)
                eval_loss += step_loss
                step += 1
            except StopIteration:
                break

        for line, p in zip(lines, pred_labels):
            res = line.strip() + "\t" + str(p) + "\n"
            pred_f.write(res)
    pred_f.close()

    eval_loss /= step

    eval_summary2, acc, rec, pre, auc = eval_sess.run([
                                        loaded_eval_model.eval_summary2,
                                        loaded_eval_model.accuracy,
                                        loaded_eval_model.recall,
                                        loaded_eval_model.precision,
                                        loaded_eval_model.auc])
    f1 = (2 * rec * pre) / (rec + pre + 0.00000001)
    scores = {"accuracy": acc,
              "recall": rec,
              "precision": pre,
              "f1": f1,
              "auc": auc}

    logging.info("# eval loss %.4f acc %.4f rec %.4f pre %.4f f1 %.4f auc %.4f" % (eval_loss, acc, rec, pre, f1, auc))

    summary_writer.add_summary(eval_summary1, global_step=global_step)
    summary_writer.add_summary(eval_summary2, global_step=global_step)
    summary_writer.close()

    for metric in config.metrics.split(","):
        best_metric_label = "best_%s" % metric

        if save_on_best and scores[metric] > getattr(config, best_metric_label):
            setattr(config, best_metric_label, scores[metric])
            loaded_eval_model.saver.save(
                eval_sess,
                os.path.join(
                    getattr(config, best_metric_label + "_dir"), ckpt_name),
                loaded_eval_model.global_step)

    return scores, global_step


def run_test(config, eval_model, eval_sess, data_file, model_dir):
    output_file = "output_" + os.path.split(data_file)[-1].split(".")[0]
    pred_file = os.path.join(model_dir, output_file)
    logger.info("  predictions to output %s." % pred_file)

    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            eval_model.model, model_dir, eval_sess, "eval")

        # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        # running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        # TODO: tf.metrics
        # eval_sess.run(running_vars_initializer)
        eval_sess.run(tf.local_variables_initializer())

    eval_data = data_helper.load_data(data_file, config.word_vocab_file, config.char_vocab_file,
                                      w_max_len1=config.max_word_len1,
                                      w_max_len2=config.max_word_len2,
                                      c_max_len1=config.max_char_len1,
                                      c_max_len2=config.max_char_len2,
                                      text_split="|", split="\t")
    eval_iterator = data_helper.batch_iterator(eval_data, batch_size=config.batch_size, shuffle=False)

    step = 0
    pred_labels = []
    lines = open(data_file, "r", encoding="utf-8").readlines()
    with open(pred_file, mode="w", encoding="utf-8") as pred_f:
        pred_f.write("")
        while True:
            try:
                b_word_ids1, b_word_ids2, b_word_len1, b_word_len2, b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels = next(
                    eval_iterator)
                pred, step_loss, acc_op, rec_op, pre_op, auc_op = \
                    loaded_eval_model.eval(eval_sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2,
                                           b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels)
                pred_labels.extend(pred)
                step += 1
            except StopIteration:
                break

        for line, p in zip(lines, pred_labels):
            res = line.strip() + "\t" + str(p) + "\n"
            pred_f.write(res)
    pred_f.close()


def test(config, model_creator):
    for metric in config.metrics.split(","):
        best_metric_label = "best_" + metric
        model_dir = getattr(config, best_metric_label + "_dir")

        logger.info("Start evaluating saved best model on training-set.")
        eval_model = model_helper.create_model(model_creator, config, mode="eval")
        session_config = utils.get_config_proto()
        eval_sess = tf.Session(config=session_config, graph=eval_model.graph)
        run_test(config, eval_model, eval_sess, config.train_file, model_dir)

        logger.info("Start evaluating saved best model on dev-set.")
        eval_model = model_helper.create_model(model_creator, config, mode="eval")
        session_config = utils.get_config_proto()
        eval_sess = tf.Session(config=session_config, graph=eval_model.graph)
        run_test(config, eval_model, eval_sess, config.dev_file, model_dir)

        logger.info("Start evaluating saved best model on test-set.")
        eval_model = model_helper.create_model(model_creator, config, mode="eval")
        session_config = utils.get_config_proto()
        eval_sess = tf.Session(config=session_config, graph=eval_model.graph)
        run_test(config, eval_model, eval_sess, config.test_file, model_dir)


def train(config, model_creator):
    steps_per_stats = config.steps_per_stats
    steps_per_eval = config.steps_per_eval
    model_dir = config.model_dir
    ckpt_name = config.ckpt_name
    ckpt_path = os.path.join(model_dir, ckpt_name)

    # Create model
    train_model = model_helper.create_model(model_creator, config, "train")
    eval_model = model_helper.create_model(model_creator, config, "eval")
    # infer_model = model_helper.create_model(model_creator, config, "infer")

    train_data = data_helper.load_data(config.train_file, config.word_vocab_file, config.char_vocab_file,
                                       w_max_len1=config.max_word_len1,
                                       w_max_len2=config.max_word_len2,
                                       c_max_len1=config.max_char_len1,
                                       c_max_len2=config.max_char_len2,
                                       text_split="|", split="\t")
    train_iterator = data_helper.batch_iterator(train_data, batch_size=config.batch_size, shuffle=True)

    eval_data = data_helper.load_data(config.dev_file, config.word_vocab_file, config.char_vocab_file,
                                      w_max_len1=config.max_word_len1,
                                      w_max_len2=config.max_word_len2,
                                      c_max_len1=config.max_char_len1,
                                      c_max_len2=config.max_char_len2,
                                      text_split="|", split="\t")
    # eval_iterator = data_helper.batch_iterator(eval_data, batch_size=config.batch_size, shuffle=False)

    # TensorFlow model
    session_config = utils.get_config_proto()
    train_sess = tf.Session(config=session_config, graph=train_model.graph)
    eval_sess = tf.Session(config=session_config, graph=eval_model.graph)
    # infer_sess = tf.Session(config=config, graph=infer_model.graph)

    # Summary Writer
    summary_writer = tf.summary.FileWriter(os.path.join(model_dir, "train_log"), train_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, model_dir, train_sess, "train")
        local_initializer = tf.local_variables_initializer()

        # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        # running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    step_time, train_loss, train_acc, train_rec, train_pre, train_f1, train_auc, gN = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    lr = loaded_train_model.learning_rate.eval(session=train_sess)
    last_stat_step = global_step
    last_eval_step = global_step

    logger.info("# Start step %d" % global_step)

    epoch_idx = 0
    while global_step < config.num_train_steps:
        start_time = time.time()
        try:
            # TODO: tf.metrics
            # train_sess.run(running_vars_initializer)
            train_sess.run(local_initializer)

            batch = next(train_iterator)
            b_word_ids1, b_word_ids2, b_word_len1, b_word_len2, b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels = batch
            # for b in batch:
            #     print(b)
            train_summary1, pred, step_loss, _, acc_op, rec_op, pre_op, auc_op, global_step, grad_norm, lr, sim = \
                loaded_train_model.train(train_sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2,
                                         b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels)
            train_summary2, step_acc, step_rec, step_pre, step_auc = \
                train_sess.run([loaded_train_model.train_summary2,
                                loaded_train_model.accuracy,
                                loaded_train_model.recall,
                                loaded_train_model.precision,
                                loaded_train_model.auc])
            config.epoch_step += 1

        except StopIteration:
            # Finished going through the training dataset.  Go to next epoch.
            epoch_idx += 1
            config.epoch_step = 0
            train_iterator = data_helper.batch_iterator(train_data, batch_size=config.batch_size, shuffle=True)
            continue

        step_time += (time.time() - start_time)
        train_loss += step_loss
        train_acc += step_acc
        train_rec += step_rec
        train_pre += step_pre
        train_auc += step_auc
        gN += grad_norm

        if global_step - last_stat_step >= steps_per_stats:
            last_stat_step = global_step
            step_time /= steps_per_stats
            train_loss /= steps_per_stats
            train_acc /= steps_per_stats
            train_rec /= steps_per_stats
            train_pre /= steps_per_stats
            train_f1 = (2 * train_rec * train_pre) / (train_rec + train_pre + 0.00000001)
            gN /= steps_per_stats

            logger.info(
                "  step %d lr %g step_time %.2fs loss %.4f acc %.4f rec %.4f pre %.4f f1 %.4f auc %.4f gN %.2f" %
                (global_step, lr, step_time, train_loss, train_acc, train_rec, train_pre, train_f1, train_auc,
                 grad_norm))
            summary_writer.add_summary(train_summary1, global_step=global_step)
            summary_writer.add_summary(train_summary2, global_step=global_step)
            step_time, train_loss, train_acc, train_rec, train_pre, train_f1, train_auc, gN = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step
            # Save checkpoint
            loaded_train_model.saver.save(train_sess, ckpt_path, global_step=global_step)
            # Evaluate on dev
            run_eval(config, eval_model, eval_sess, eval_data, model_dir, ckpt_name=ckpt_name, save_on_best=True)

    logger.info("# Finished epoch %d, step %d." % (epoch_idx, global_step))

    # Done training
    loaded_train_model.saver.save(train_sess, ckpt_path, global_step=global_step)
    logger.info("# Final, step %d lr %g step_time %.2fs loss %.4f acc %.4f rec %.4f pre %.4f f1 %.4f auc %.4f gN %.2f" %
                (global_step, lr, step_time, train_loss, train_acc, train_rec, train_pre, train_f1, train_auc, gN))
    logger.info("# Done training!")

    summary_writer.close()
