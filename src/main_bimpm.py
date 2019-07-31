#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-07-11
Author: aitingliu
"""
import logging

import tensorflow as tf

from utils import misc_utils
import train
import inference
from models import BiMPM

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
flags.DEFINE_string("model_dir", "/ceph/qbkg/aitingliu/qq/src/model/BiMPM/", "model path")
flags.DEFINE_string("log_dir", "/ceph/qbkg/aitingliu/qq/src/logdir/BiMPM/", "logdir path")

# data file path
flags.DEFINE_string("train_file", "/ceph/qbkg/aitingliu/qq/data/20190726/raw/train.txt", "Training data file.")
flags.DEFINE_string("dev_file", "/ceph/qbkg/aitingliu/qq/data/20190726/raw/dev.txt", "Development data file.")
flags.DEFINE_string("test_file", "/ceph/qbkg/aitingliu/qq/data/20190726/raw/test.txt", "Test data file.")
flags.DEFINE_string("infer_file", "/ceph/qbkg/aitingliu/qq/data/20190726/raw/test.txt", "Test data file.")
flags.DEFINE_string("word_vocab_file", "/ceph/qbkg/aitingliu/qq/data/20190726/raw/word.txt", "Word vocabulary file.")
flags.DEFINE_string("char_vocab_file", "/ceph/qbkg/aitingliu/qq/data/20190726/raw/char.txt", "Char vocabulary file.")
flags.DEFINE_string("word_embed_file", None, "Pretrained embedding file.")
flags.DEFINE_string("ckpt_name", "model.ckpt", "Checkpoint file name.")

# train
flags.DEFINE_integer("random_seed", 1213, "Random seed (>0, set a specific seed).")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate. Adam: 0.001 | 0.0001")
flags.DEFINE_string("opt", "adam", "Optimizer: adam | adadelta | adagrad | sgd | momentum | rmsprop")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 128, "Batch size. (default: 128)")
flags.DEFINE_integer("epoch_step", 0, "Record where we were within an epoch.")
flags.DEFINE_integer("num_train_epochs", 50, "Num epochs to train.")
flags.DEFINE_integer("num_keep_ckpts", 5, "Max number fo checkpoints to keep.")
flags.DEFINE_integer("num_train_steps", 20000, "Num epochs to train.")
flags.DEFINE_integer("steps_per_stats", 100, "How many training steps to do per stats logging.")
flags.DEFINE_integer("steps_per_eval", 1000, "How many training steps to do per evaluation.")
flags.DEFINE_string("metrics", "accuracy,f1,eval_loss", "evaluations metrics (accuracy | f1)")
flags.DEFINE_float("best_accuracy", 0.0, "Best accuracy score on dev set")
flags.DEFINE_string("best_accuracy_dir", None, "Best accuracy model dir")
flags.DEFINE_float("best_f1", 0.0, "Best f1 score on dev set")
flags.DEFINE_string("best_f1_dir", None, "Best f1 model dir")
flags.DEFINE_float("best_eval_loss", 1000.0, "Best eval loss on dev set.")
flags.DEFINE_string("best_eval_loss_dir", None, "Best eval loss on dev set.")

# inference
flags.DEFINE_integer("infer_batch_size", 64, "Batch size for inference mode.")

# data constraint
flags.DEFINE_integer("max_word_len1", 10, "Max length of sent1 length in word level.")
flags.DEFINE_integer("max_word_len2", 10, "Max length of sent2 length in word level.")
flags.DEFINE_integer("max_char_len1", 20, "Max length of sent1 length in char level.")
flags.DEFINE_integer("max_char_len2", 20, "Max length of sent2 length in char level.")
flags.DEFINE_integer("word_embed_size", 300, "Word embedding size.")
flags.DEFINE_integer("char_embed_size", 300, "Char embedding size.")
flags.DEFINE_integer("word_vocab_size", 30000, "Word vocabulary size.")
flags.DEFINE_integer("char_vocab_size", 4000, "Char vocabulary size.")

# model configuration
# flags.DEFINE_integer("num_units", 128, "LSTM hidden size.(default: 128)")
# flags.DEFINE_string("unit_type", "lstm", "RNN type: lstm | gru | layer_norm_lstm")
# flags.DEFINE_string("filter_sizes", "3,4,5", "CNN filter sizes.")
# flags.DEFINE_integer("num_filters", 100, "Number of filters.")
flags.DEFINE_integer("fc_size", 512, "Fully connected layer hidden size. (default: 1024)")
flags.DEFINE_integer("num_classes", 2, "Number of classes.")
flags.DEFINE_float("dropout_rate", 0.8, "Dropout rate (not keep_prob) default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_integer("decay_steps", 1000, "How many steps before decay learning rate. (default: 500)")
flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")

flags.DEFINE_integer("context_layer_num", 1, "")
flags.DEFINE_integer("context_lstm_dim", 100, "Number of dimension for context representation layer.")
flags.DEFINE_integer("aggregation_layer_num", 1, "")
flags.DEFINE_integer("aggregation_lstm_dim", 100, "")

flags.DEFINE_boolean("with_full_match", True, "With full matching")
flags.DEFINE_boolean("with_maxpool_match", True, "With maxpooling matching")
flags.DEFINE_boolean("with_attentive_match", True, "With attentive matching")
flags.DEFINE_boolean("with_max_attentive_match", True, "With max attentive matching")

flags.DEFINE_boolean("with_cosine", True, "")
flags.DEFINE_boolean("with_mp_cosine", True, "")
flags.DEFINE_integer("cosine_MP_dim", 5, "")

flags.DEFINE_integer("att_dim", 50, "Number of dimension for context representation layer.")
flags.DEFINE_string("att_type", "symmetric", "Number of dimension for context representation layer.")

flags.DEFINE_integer("highway_layer_num", 1, "")
flags.DEFINE_boolean("with_highway", False, "")
flags.DEFINE_boolean("with_match_highway", False, "")
flags.DEFINE_boolean("with_aggregation_highway", False, "")

flags.DEFINE_boolean("use_cudnn", False, "")

# train/inference/evaluate flag
flags.DEFINE_boolean("train", True, "train mode")
flags.DEFINE_boolean("infer", False, "infer mode")
flags.DEFINE_boolean("test", True, "test mode")


def main(unused):

    misc_utils.update_config(FLAGS)

    model_creator = BiMPM.BiMPM

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
