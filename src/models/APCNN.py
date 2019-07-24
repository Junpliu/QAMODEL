#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-07-11
Author: aitingliu
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging

import tensorflow as tf

from utils import model_helper
from utils import misc_utils

logger = logging.getLogger(__name__)


class APCNN(object):
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
        self.filter_sizes = [int(x) for x in config.filter_sizes.split(",")]
        self.num_filters = config.num_filters
        self.fc_size = config.fc_size
        self.num_classes = config.num_classes
        self.dropout = config.dropout if mode == "train" else 0.0
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

    def textcnn(self, inputs, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            # Create a convolution layer for each filter size
            outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                conv = tf.layers.conv1d(inputs,
                                        self.num_filters,
                                        filter_size,
                                        padding="same",
                                        name='conv1d_%d' % filter_size)  # (batch_size, seq_length，num_filters)
                outputs.append(conv)

            # Combine all the features
            outputs = tf.concat(outputs, 2, name="output")  # (batch_size, seq_length, num_filters_total)
            # TODO: whether use dropout there
            # if self.mode == "train":
            #     outputs = tf.nn.dropout(outputs, keep_prob=(1.0 - self.dropout), name="output")

        return outputs

    def attentive_pooling(self, rep1, rep2, reuse, name):
        vector_size = rep1.get_shape().as_list()[-1]
        seq1_len = rep1.get_shape().as_list()[1]

        with tf.variable_scope("%s/attentive-pooling" % name, reuse=reuse):
            # TODO: U should be [vector_size, vector_size]
            U = tf.get_variable("U", shape=[vector_size, vector_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            # U_batch = tf.tile(tf.expand_dims(U, 0), [self.batch_size, 1, 1])
            # self.U_batch = U_batch
            # G = tf.tanh(
            #     tf.matmul(
            #         tf.matmul(rep1, U_batch), tf.transpose(rep2, [0, 2, 1])))  # (batch_size, seq1_len, seq2_len)
            G = tf.tanh(
                tf.einsum("bme,ben->bmn",
                          tf.reshape(
                              tf.matmul(tf.reshape(rep1, shape=[-1, vector_size]), U),
                              shape=[-1, seq1_len, vector_size]),
                          tf.transpose(rep2, [0, 2, 1])))

            # column-wise and row-wise max-pooling
            # to generate g_rep1 (batch_size*seq1_len*1), g_rep2 (batch_size*1*seq2_len)
            g_rep1 = tf.reduce_max(G, axis=2, keepdims=True)
            g_rep2 = tf.reduce_max(G, axis=1, keepdims=True)

            # create attention vectors sigma_rep1 (batch_size*seq1_len), sigma_rep2 (batch_size*seq2_len)
            sigma_rep1 = tf.nn.softmax(g_rep1)
            sigma_rep2 = tf.nn.softmax(g_rep2)

            # final output r_rep1, r_rep2 (batch_size*vector_size)
            r_rep1 = tf.squeeze(tf.matmul(tf.transpose(rep1, [0, 2, 1]), sigma_rep1), axis=2)
            r_rep2 = tf.squeeze(tf.matmul(sigma_rep2, rep2), axis=1)

            return r_rep1, r_rep2

    def build_graph(self):
        with tf.variable_scope("net"):
            self.init_embedding()
            word_embed1 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids1, "word_embed1")
            word_embed2 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids2, "word_embed2")
            char_embed1 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids1, "char_embed1")
            char_embed2 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids2, "char_embed2")

            # TODO: query/question share cnn weights.
            word_rep1 = self.textcnn(word_embed1, tf.AUTO_REUSE, "word")
            word_rep2 = self.textcnn(word_embed2, tf.AUTO_REUSE, "word")

            char_rep1 = self.textcnn(char_embed1, tf.AUTO_REUSE, "char")
            char_rep2 = self.textcnn(char_embed2, tf.AUTO_REUSE, "char")

            word_reps_seq1, word_reps_seq2 = self.attentive_pooling(word_rep1, word_rep2, tf.AUTO_REUSE, "word")
            char_reps_seq1, char_reps_seq2 = self.attentive_pooling(char_rep1, char_rep2, tf.AUTO_REUSE, "char")

            sent_merge1 = tf.concat([word_reps_seq1, char_reps_seq1], axis=1, name="sent_merge1")
            sent_merge2 = tf.concat([word_reps_seq2, char_reps_seq2], axis=1, name="sent_merge2")

            reps_cat = tf.concat([sent_merge1, sent_merge2], axis=1)
            reps_add = tf.add(sent_merge1, sent_merge2)
            reps_sub = tf.subtract(sent_merge1, sent_merge2)
            reps_abs_sub = tf.abs(tf.subtract(sent_merge1, sent_merge2))
            reps_mul = tf.multiply(sent_merge1, sent_merge2)
            reps_match = tf.concat([reps_cat, reps_add, reps_sub, reps_abs_sub, reps_mul], axis=1)

            sent_dense1 = tf.layers.dense(inputs=reps_match, units=self.fc_size, activation=tf.nn.relu)
            sent_dense1 = tf.nn.dropout(sent_dense1, keep_prob=(1.0 - self.dropout))
            final_out = tf.layers.dense(inputs=sent_dense1, units=self.num_classes)

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
