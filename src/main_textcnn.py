#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-07-11
Author: aitingliu
"""
import os
import random
import logging

import tensorflow as tf
import numpy as np

from utils import vocab_utils
import train
import inference
from models import TextCNN

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flags = tf.flags
FLAGS = flags.FLAGS

# save model path
flags.DEFINE_string("model_dir", "/ceph/qbkg/aitingliu/qq/src/model/qq_simscore/TextCNN", "model path")

# data file path
flags.DEFINE_string("train_file", "/ceph/qbkg/aitingliu/qq/src/data/qq_simscore/train.txt", "Training data file.")
flags.DEFINE_string("dev_file", "/ceph/qbkg/aitingliu/qq/src/data/qq_simscore/dev.txt", "Development data file.")
flags.DEFINE_string("test_file", "/ceph/qbkg/aitingliu/qq/src/data/qq_simscore/test.txt", "Test data file.")
flags.DEFINE_string("infer_file", "/ceph/qbkg/aitingliu/qq/src/data/qq_simscore/test.txt", "Test data file.")
flags.DEFINE_string("word_vocab_file", "/ceph/qbkg/aitingliu/qq/src/data/qq_simscore/word.txt", "Word vocabulary file.")
flags.DEFINE_string("char_vocab_file", "/ceph/qbkg/aitingliu/qq/src/data/qq_simscore/char.txt", "Char vocabulary file.")
flags.DEFINE_string("word_embed_file", None, "Pretrained embedding file.")
flags.DEFINE_string("ckpt_name", "textcnn.ckpt", "Checkpoint file name.")

# train
flags.DEFINE_integer("random_seed", 1213, "Random seed (>0, set a specific seed).")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate. Adam: 0.001 | 0.0001")
flags.DEFINE_string("opt", "adam", "Optimizer: adam | adadelta | adagrad | sgd | momentum | rmsprop")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 128, "Batch size. (default: 128)")
flags.DEFINE_integer("epoch_step", 0, "Record where we were within an epoch.")
flags.DEFINE_integer("num_train_epochs", 20, "Num epochs to train.")
flags.DEFINE_integer("num_keep_ckpts", 5, "Max number fo checkpoints to keep.")
flags.DEFINE_integer("num_train_steps", 20000, "Num epochs to train.")
flags.DEFINE_integer("steps_per_stats", 100, "How many training steps to do per stats logging.")
flags.DEFINE_integer("steps_per_eval", 1000, "How many training steps to do per evaluation.")
flags.DEFINE_string("metrics", "accuracy,f1", "evaluations metrics (accuracy | f1)")
flags.DEFINE_float("best_accuracy", 0.0, "Best accuracy score on dev set")
flags.DEFINE_string("best_accuracy_dir", None, "Best accuracy model dir")
flags.DEFINE_float("best_f1", 0.0, "Best f1 score on dev set")
flags.DEFINE_string("best_f1_dir", None, "Best f1 model dir")

# inference
flags.DEFINE_integer("infer_batch_size", 64, "Batch size for inference mode.")

# data constraint
flags.DEFINE_integer("max_word_len1", 40, "Max length of sent1 length in word level.")
flags.DEFINE_integer("max_word_len2", 40, "Max length of sent2 length in word level.")
flags.DEFINE_integer("max_char_len1", 40, "Max length of sent1 length in char level.")
flags.DEFINE_integer("max_char_len2", 40, "Max length of sent2 length in char level.")
flags.DEFINE_integer("word_embed_size", 300, "Word embedding size.")
flags.DEFINE_integer("char_embed_size", 300, "Char embedding size.")
flags.DEFINE_integer("word_vocab_size", 30000, "Word vocabulary size.")
flags.DEFINE_integer("char_vocab_size", 4000, "Char vocabulary size.")

# bilstm model configuration

# flags.DEFINE_integer("num_units", 128, "LSTM hidden size.(default: 128)")
# flags.DEFINE_string("unit_type", "lstm", "RNN type: lstm | gru | layer_norm_lstm")
flags.DEFINE_string("filter_sizes", "3,4,5", "CNN filter sizes.")
flags.DEFINE_integer("num_filters", 100, "Number of filters.")
flags.DEFINE_integer("fc_size", 512, "Fully connected layer hidden size. (default: 1024)")
flags.DEFINE_integer("num_classes", 2, "Number of classes.")
flags.DEFINE_float("dropout", 0.3, "Dropout rate (not keep_prob) default: 0.3)")
flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# train/inference/evaluate flag
flags.DEFINE_boolean("train", True, "train mode")
flags.DEFINE_boolean("infer", False, "infer mode")
flags.DEFINE_boolean("test", True, "test mode")


def update_config(config):
    # Vocab
    vocab_utils.create_vocab(config.train_file, config.word_vocab_file, split="|", char_level=False)
    vocab_utils.create_vocab(config.train_file, config.char_vocab_file, split="|", char_level=True)
    word_vocab, word_vocab_size = vocab_utils.load_vocab(config.word_vocab_file)
    char_vocab, char_vocab_size = vocab_utils.load_vocab(config.char_vocab_file)
    logger.info("# Updating config.word_vocab_size: {} -> {}".format(str(config.word_vocab_size), str(word_vocab_size)))
    logger.info("# Updating config.char_vocab_size: {} -> {}".format(str(config.char_vocab_size), str(char_vocab_size)))
    config.word_vocab_size = word_vocab_size
    config.char_vocab_size = char_vocab_size

    # Pretrained Embeddings
    if config.word_embed_file:
        embed_dict, word_embed_size = vocab_utils.load_embed_txt(config.word_embed_file)
        logger.info("# Updating config.embed_size: {} -> {}".format(str(config.word_embed_size), str(word_embed_size)))
        config.word_embed_size = word_embed_size

    # Model output directory
    model_dir = config.model_dir
    if model_dir and not os.path.exists(model_dir):
        logger.info("# Creating model directory %s ..." % model_dir)
        os.makedirs(model_dir)

    # Evaluation
    config.best_accuracy = .0
    best_accuracy_dir = os.path.join(config.model_dir, "best_accuracy")
    setattr(config, "best_accuracy_dir", best_accuracy_dir)
    if best_accuracy_dir and not os.path.exists(best_accuracy_dir):
        os.makedirs(best_accuracy_dir)

    config.best_f1 = .0
    best_f1_dir = os.path.join(config.model_dir, "best_f1")
    setattr(config, "best_f1_dir", best_f1_dir)
    if best_f1_dir and not os.path.exists(best_f1_dir):
        os.makedirs(best_f1_dir)

    # print configuration
    logger.info("# Hparams")
    # for arg in vars(config):
    #     logger.info("  {}\t{}".format(arg, getattr(config, arg)))
    for arg, value in FLAGS.__flags.items():
        logger.info("  {}\t{}".format(arg, str(value.value)))


def main(unused):

    update_config(FLAGS)

    # Random
    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        logger.info("# Set random seed to %d" % random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    model_creator = TextCNN.TextCNN

    # train
    if FLAGS.train:
        logger.info("TRAIN")
        train.train(FLAGS, model_creator)

    # evaluate
    if FLAGS.test:
        logger.info("TEST")
        train.test(FLAGS, model_creator)

    # inference
    if FLAGS.infer:
        logger.info("INFER")
        inference.inference(FLAGS, model_creator)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # add_arguments(parser)
    # config, unparsed = parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run()
