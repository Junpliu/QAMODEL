#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-07-22
Author: aitingliu
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging

import tensorflow as tf

from utils import model_helper
from utils import misc_utils
from . import match_utils

logger = logging.getLogger(__name__)


class BiMPM(object):
    def __init__(self,
                 config,
                 mode="train"):
        self.config = config
        self.mode = mode
        self.dtype = tf.float32

        self.word_vocab_file = config.word_vocab_file
        self.char_vocab_file = config.char_vocab_file
        self.word_embed_file = config.word_embed_file

        self.word_vocab_size = config.word_vocab_size
        self.char_vocab_size = config.char_vocab_size
        self.word_embed_size = config.word_embed_size
        self.char_embed_size = config.char_embed_size
        self.max_word_len1 = config.max_word_len1
        self.max_word_len2 = config.max_word_len2
        self.max_char_len1 = config.max_char_len1
        self.max_char_len2 = config.max_char_len2

        self.random_seed = config.random_seed
        # self.unit_type = config.unit_type
        # self.num_units = config.num_units
        # self.filter_sizes = [int(x) for x in config.filter_sizes.split(",")]
        # self.num_filters = config.num_filters
        self.fc_size = config.fc_size
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate if mode == "train" else 0.0
        self.l2_reg_lambda = config.l2_reg_lambda
        self.learning_rate = config.learning_rate
        self.opt = config.opt
        self.max_gradient_norm = config.max_gradient_norm
        self.num_keep_ckpts = config.num_keep_ckpts
        self.batch_size = config.batch_size
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate

        self.global_step = tf.Variable(0, trainable=False)
        self.build()

    def init_embedding(self):
        """Init embedding."""
        with tf.variable_scope("embed"):
            self.word_embedding, self.word_embed_size = \
                model_helper.create_or_load_embed("word_embedding",
                                                  self.word_vocab_file,
                                                  self.word_embed_file,
                                                  self.word_vocab_size,
                                                  self.word_embed_size,
                                                  dtype=self.dtype,
                                                  trainable=True,
                                                  seed=self.random_seed)
            self.char_embedding, self.char_embed_size = \
                model_helper.create_or_load_embed("char_embedding",
                                                  self.char_vocab_file,
                                                  None,
                                                  self.char_vocab_size,
                                                  self.char_embed_size,
                                                  dtype=self.dtype,
                                                  trainable=True,
                                                  seed=self.random_seed)

    def add_placeholders(self):
        with tf.variable_scope("inputs"):
            self.word_ids1 = tf.placeholder(tf.int32, [None, self.max_word_len1], "p_word_ids1")
            self.word_ids2 = tf.placeholder(tf.int32, [None, self.max_word_len2], "p_word_ids2")
            self.word_len1 = tf.placeholder(tf.int32, [None], "p_word_len1")
            self.word_len2 = tf.placeholder(tf.int32, [None], "p_word_len2")

            self.char_ids1 = tf.placeholder(tf.int32, [None, self.max_char_len1], "p_char_ids1")
            self.char_ids2 = tf.placeholder(tf.int32, [None, self.max_char_len2], "p_char_ids2")
            self.char_len1 = tf.placeholder(tf.int32, [None], "p_char_len1")
            self.char_len2 = tf.placeholder(tf.int32, [None], "p_char_len2")

            if self.mode != "infer":
                self.labels = tf.placeholder(tf.int32, [None], "labels")
                self.target = tf.one_hot(self.labels, depth=self.num_classes, dtype=self.dtype)

    def compute_mean_vector(self, embed_seq, max_seq_len):
        with tf.variable_scope("compute_mean"):
            mean_rep = tf.div(tf.reduce_sum(embed_seq, axis=1), max_seq_len)
        return mean_rep

    def build_graph(self):
        with tf.variable_scope("net"):
            # ================== word representation layer ==================
            self.init_embedding()
            # TODO: add word level representation
            word_embed1 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids1, "word_embed1")
            word_embed2 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids2, "word_embed2")
            word_mask_seq1 = tf.sequence_mask(self.word_len1, self.max_word_len1, dtype=tf.float32)
            word_mask_seq2 = tf.sequence_mask(self.word_len2, self.max_word_len2, dtype=tf.float32)
            word_input_dim = self.word_embed_size
            
            char_embed1 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids1, "char_embed1")
            char_embed2 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids2, "char_embed2")
            char_mask_seq1 = tf.sequence_mask(self.char_len1, self.max_char_len1, dtype=tf.float32)
            char_mask_seq2 = tf.sequence_mask(self.char_len2, self.max_char_len2, dtype=tf.float32)
            char_input_dim = self.char_embed_size
            is_training = True if self.mode == "train" else False

            # ================== Bilateral Matching ==================

            with tf.variable_scope("word"):
                (word_match_representation, word_match_dim) = match_utils.bilateral_match_func(
                    word_embed1, word_embed2, self.word_len1, self.word_len2,
                    word_mask_seq1, word_mask_seq2, word_input_dim, is_training, options=self.config)
            with tf.variable_scope("char"):
                (char_match_representation, char_match_dim) = match_utils.bilateral_match_func(
                    char_embed1, char_embed2, self.char_len1, self.char_len2,
                    char_mask_seq1, char_mask_seq2, char_input_dim, is_training, options=self.config)

            match_representation = tf.concat([word_match_representation, char_match_representation], axis=-1)
            match_dim = word_match_dim + char_match_dim

            # ================== Prediction Layer ====================
            sent_dense = tf.layers.dense(inputs=match_representation, units=match_dim//2, activation=tf.nn.relu)
            sent_dense = tf.nn.dropout(sent_dense, keep_prob=(1.0 - self.dropout_rate))
            final_out = tf.layers.dense(inputs=sent_dense, units=self.num_classes)

            self.logits = final_out

    def add_loss_op(self):
        with tf.variable_scope("indicators"):
            self.predictions = tf.nn.softmax(self.logits)
            self.pred_labels = tf.argmax(self.predictions, axis=1, output_type=tf.int32)
            self.simscore = self.predictions[:, -1]

        if self.mode != "infer":
            with tf.variable_scope("loss"):
                self.losses = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits))
                if self.l2_reg_lambda > 0.0:
                    self.l2_losses = tf.add_n(
                        [tf.nn.l2_loss(tf.cast(v, self.dtype)) for v in tf.trainable_variables()],
                        name="l2_losses")
                    self.losses = self.losses + self.l2_losses * self.l2_reg_lambda

            with tf.variable_scope("metrics"):
                self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.pred_labels, name="acc")
                self.recall, self.recall_op = tf.metrics.recall(self.labels, self.pred_labels, name="rec")
                self.precision, self.precision_op = tf.metrics.precision(self.labels, self.pred_labels, name="pre")
                self.auc, self.auc_op = tf.metrics.auc(self.labels, self.pred_labels, name="auc")

    def add_train_op(self):
        if self.mode == "train":
            params = tf.trainable_variables()
            with tf.variable_scope("opt"):
                self.learning_rate = tf.train.exponential_decay(
                    learning_rate=self.learning_rate,
                    global_step=self.global_step,
                    decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate,
                    staircase=True,
                    name="learning_rate_decay")
                if self.opt == 'adam':
                    opt = tf.train.AdamOptimizer(self.learning_rate)
                elif self.opt == 'adagrad':
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                elif self.opt == "adadelta":
                    opt = tf.train.AdadeltaOptimizer(self.learning_rate)
                elif self.opt == 'sgd':
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                elif self.opt == 'momentum':
                    opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
                elif self.opt == 'rmsprop':
                    opt = tf.train.RMSPropOptimizer(self.learning_rate)
                else:
                    raise NotImplementedError("Unknown method {}".format(self.opt))
                gradients = tf.gradients(self.losses, params)
                self.clipped_gradients, self.grad_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = opt.apply_gradients(zip(self.clipped_gradients, params), global_step=self.global_step)

    def add_summary(self):
        if self.mode == "train":
            self.train_summary1 = tf.summary.merge([tf.summary.scalar("loss", self.losses),
                                                    tf.summary.scalar("learning_rate", self.learning_rate),
                                                    tf.summary.scalar("grad_norm", self.grad_norm),
                                                    tf.summary.scalar("clipped_gradient",
                                                                      tf.global_norm(self.clipped_gradients))])
            self.train_summary2 = tf.summary.merge([tf.summary.scalar("accuracy", self.accuracy),
                                                    tf.summary.scalar("recall", self.recall),
                                                    tf.summary.scalar("precision", self.precision),
                                                    tf.summary.scalar("auc", self.auc)])
        elif self.mode == "eval":
            self.eval_summary1 = tf.summary.merge([tf.summary.scalar("loss", self.losses)])
            self.eval_summary2 = tf.summary.merge([tf.summary.scalar("accuracy", self.accuracy),
                                                   tf.summary.scalar("recall", self.recall),
                                                   tf.summary.scalar("precision", self.precision),
                                                   tf.summary.scalar("auc", self.auc)])

    def add_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_keep_ckpts)

    def build(self):
        self.add_placeholders()
        self.build_graph()
        self.add_loss_op()
        self.add_train_op()
        self.add_summary()
        self.add_saver()
        misc_utils.print_params()

    def train(self, sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2,
              b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels):
        input_feed = {
            self.word_ids1: b_word_ids1,
            self.word_ids2: b_word_ids2,
            self.word_len1: b_word_len1,
            self.word_len2: b_word_len2,

            self.char_ids1: b_char_ids1,
            self.char_ids2: b_char_ids2,
            self.char_len1: b_char_len1,
            self.char_len2: b_char_len2,

            self.labels: b_labels}

        output_feed = [self.train_summary1,
                       self.simscore, self.losses,
                       self.train_op, self.accuracy_op, self.recall_op, self.precision_op, self.auc_op,
                       self.global_step, self.grad_norm, self.learning_rate]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def eval(self, sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2,
             b_char_ids1, b_char_ids2, b_char_len1, b_char_len2, b_labels):
        input_feed = {
            self.word_ids1: b_word_ids1,
            self.word_ids2: b_word_ids2,
            self.word_len1: b_word_len1,
            self.word_len2: b_word_len2,

            self.char_ids1: b_char_ids1,
            self.char_ids2: b_char_ids2,
            self.char_len1: b_char_len1,
            self.char_len2: b_char_len2,

            self.labels: b_labels}
        output_feed = [self.eval_summary1,
                       self.simscore, self.losses,
                       self.accuracy_op, self.recall_op, self.precision_op, self.auc_op]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def infer(self, sess, b_word_ids1, b_word_ids2, b_word_len1, b_word_len2,
              b_char_ids1, b_char_ids2, b_char_len1, b_char_len2):
        input_feed = {
            self.word_ids1: b_word_ids1,
            self.word_ids2: b_word_ids2,
            self.word_len1: b_word_len1,
            self.word_len2: b_word_len2,

            self.char_ids1: b_char_ids1,
            self.char_ids2: b_char_ids2,
            self.char_len1: b_char_len1,
            self.char_len2: b_char_len2}
        output_feed = self.simscore
        outputs = sess.run(output_feed, input_feed)
        return outputs
