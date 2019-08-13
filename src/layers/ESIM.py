#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-08-13
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


class ESIM(object):
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
        self.num_units = config.num_units
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
        self.use_cudnn = config.use_cudnn

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

    def bilstm_layer_cudnn(self, input_data, num_layers, rnn_size, keep_prob=1.):
        """Multi-layer BiLSTM cudnn version, faster
        Args:
            input_data: float32 Tensor of shape [seq_length, batch_size, dim].
            num_layers: int64 scalar, number of layers.
            rnn_size: int64 scalar, hidden size for undirectional LSTM.
            keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers
        Return:
            output: float32 Tensor of shape [seq_length, batch_size, dim * 2]
        """
        with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=num_layers,
                num_units=rnn_size,
                input_mode="linear_input",
                direction="bidirectional",
                dropout=1 - keep_prob)

            # to do, how to include input_mask
            outputs, output_states = lstm(inputs=input_data)

        return outputs

    def bilstm_layer(self, input_data, num_layers, rnn_size, keep_prob=1.):
        """Multi-layer BiLSTM
        Args:
            input_data: float32 Tensor of shape [seq_length, batch_size, dim].
            num_layers: int64 scalar, number of layers.
            rnn_size: int64 scalar, hidden size for undirectional LSTM.
            keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers
        Return:
            output: float32 Tensor of shape [seq_length, batch_size, dim * 2]
        """
        input_data = tf.transpose(input_data, [1, 0, 2])

        output = input_data
        for layer in range(num_layers):
            with tf.variable_scope('bilstm_{}'.format(layer), reuse=tf.AUTO_REUSE):
                cell_fw = tf.contrib.rnn.LSTMCell(
                    rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(
                    rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                  cell_bw,
                                                                  output,
                                                                  dtype=tf.float32)

                # Concat the forward and backward outputs
                output = tf.concat(outputs, 2)

        output = tf.transpose(output, [1, 0, 2])

        return output

    def local_inference(self, x1, x1_mask, x2, x2_mask):
        """Local inference collected over sequences
        Args:
            x1: float32 Tensor of shape [seq_length1, batch_size, dim].
            x1_mask: float32 Tensor of shape [seq_length1, batch_size].
            x2: float32 Tensor of shape [seq_length2, batch_size, dim].
            x2_mask: float32 Tensor of shape [seq_length2, batch_size].
        Return:
            x1_dual: float32 Tensor of shape [seq_length1, batch_size, dim]
            x2_dual: float32 Tensor of shape [seq_length2, batch_size, dim]
        """

        # x1: [batch_size, seq_length1, dim].
        # x1_mask: [batch_size, seq_length1].
        # x2: [batch_size, seq_length2, dim].
        # x2_mask: [batch_size, seq_length2].
        x1 = tf.transpose(x1, [1, 0, 2])
        x1_mask = tf.transpose(x1_mask, [1, 0])
        x2 = tf.transpose(x2, [1, 0, 2])
        x2_mask = tf.transpose(x2_mask, [1, 0])

        # attention_weight: [batch_size, seq_length1, seq_length2]
        attention_weight = tf.matmul(x1, tf.transpose(x2, [0, 2, 1]))

        # calculate normalized attention weight x1 and x2
        # attention_weight_2: [batch_size, seq_length1, seq_length2]
        attention_weight_2 = tf.exp(
            attention_weight - tf.reduce_max(attention_weight, axis=2, keepdims=True))
        attention_weight_2 = attention_weight_2 * tf.expand_dims(x2_mask, 1)
        # alpha: [batch_size, seq_length1, seq_length2]
        alpha = attention_weight_2 / (tf.reduce_sum(attention_weight_2, -1, keepdims=True) + 1e-8)
        # x1_dual: [batch_size, seq_length1, dim]
        x1_dual = tf.reduce_sum(tf.expand_dims(x2, 1) * tf.expand_dims(alpha, -1), 2)
        # x1_dual: [seq_length1, batch_size, dim]
        x1_dual = tf.transpose(x1_dual, [1, 0, 2])

        # attention_weight_1: [batch_size, seq_length2, seq_length1]
        attention_weight_1 = attention_weight - tf.reduce_max(attention_weight, axis=1, keepdims=True)
        attention_weight_1 = tf.exp(tf.transpose(attention_weight_1, [0, 2, 1]))
        attention_weight_1 = attention_weight_1 * tf.expand_dims(x1_mask, 1)

        # beta: [batch_size, seq_length2, seq_length1]
        beta = attention_weight_1 / \
               (tf.reduce_sum(attention_weight_1, -1, keepdims=True) + 1e-8)
        # x2_dual: [batch_size, seq_length2, dim]
        x2_dual = tf.reduce_sum(tf.expand_dims(x1, 1) * tf.expand_dims(beta, -1), 2)
        # x2_dual: [seq_length2, batch_size, dim]
        x2_dual = tf.transpose(x2_dual, [1, 0, 2])

        return x1_dual, x2_dual

    def esim(self, emb1, x1_mask, emb2, x2_mask, name, reuse):

        with tf.variable_scope(name, reuse=reuse):
            # encode the sentence pair
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                if self.use_cudnn:
                    x1_enc = self.bilstm_layer_cudnn(emb1, 1, self.num_units)
                    x2_enc = self.bilstm_layer_cudnn(emb2, 1, self.num_units)
                else:
                    x1_enc = self.bilstm_layer(emb1, 1, self.num_units)
                    x2_enc = self.bilstm_layer(emb2, 1, self.num_units)

            x1_enc = x1_enc * tf.expand_dims(x1_mask, -1)
            x2_enc = x2_enc * tf.expand_dims(x2_mask, -1)

            # local inference modeling based on attention mechanism
            x1_dual, x2_dual = self.local_inference(x1_enc, x1_mask, x2_enc, x2_mask)

            x1_match = tf.concat([x1_enc, x1_dual, x1_enc * x1_dual, x1_enc - x1_dual], 2)
            x2_match = tf.concat([x2_enc, x2_dual, x2_enc * x2_dual, x2_enc - x2_dual], 2)

            # mapping high dimension feature to low dimension
            with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
                x1_match_mapping = tf.layers.dense(x1_match, self.num_units,
                                                   activation=tf.nn.relu,
                                                   name="fnn",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                x2_match_mapping = tf.layers.dense(x2_match, self.num_units,
                                                   activation=tf.nn.relu,
                                                   name="fnn",
                                                   kernel_initializer=tf.truncated_normal_initializer(
                                                       stddev=0.02),
                                                   reuse=True)

            x1_match_mapping = tf.nn.dropout(x1_match_mapping, (1. - self.dropout_rate))
            x2_match_mapping = tf.nn.dropout(x2_match_mapping, (1. - self.dropout_rate))

            # inference composition
            with tf.variable_scope("composition", reuse=tf.AUTO_REUSE):
                if self.use_cudnn:
                    x1_cmp = self.bilstm_layer_cudnn(x1_match_mapping, 1, self.num_units)
                    x2_cmp = self.bilstm_layer_cudnn(x2_match_mapping, 1, self.num_units)
                else:
                    x1_cmp = self.bilstm_layer(x1_match_mapping, 1, self.num_units)
                    x2_cmp = self.bilstm_layer(x2_match_mapping, 1, self.num_units)

            logit_x1_sum = tf.reduce_sum(x1_cmp * tf.expand_dims(x1_mask, -1), 0) / \
                           tf.expand_dims(tf.reduce_sum(x1_mask, 0), 1)
            logit_x1_max = tf.reduce_max(x1_cmp * tf.expand_dims(x1_mask, -1), 0)
            logit_x2_sum = tf.reduce_sum(x2_cmp * tf.expand_dims(x2_mask, -1), 0) / \
                           tf.expand_dims(tf.reduce_sum(x2_mask, 0), 1)
            logit_x2_max = tf.reduce_max(x2_cmp * tf.expand_dims(x2_mask, -1), 0)

            logit = tf.concat([logit_x1_sum, logit_x1_max, logit_x2_sum, logit_x2_max], 1)

        return logit

    def build_graph(self):
        with tf.variable_scope("net"):
            # ================== word representation layer ==================
            self.init_embedding()
            # TODO: add word level representation
            word_x1_enc = tf.nn.embedding_lookup(self.word_embedding, self.word_ids1, "word_embed1")
            word_x2_enc = tf.nn.embedding_lookup(self.word_embedding, self.word_ids2, "word_embed2")
            word_x1_mask = tf.sequence_mask(self.word_len1, self.max_word_len1, dtype=tf.float32)
            word_x2_mask = tf.sequence_mask(self.word_len2, self.max_word_len2, dtype=tf.float32)

            char_x1_enc = tf.nn.embedding_lookup(self.char_embedding, self.char_ids1, "char_embed1")
            char_x2_enc = tf.nn.embedding_lookup(self.char_embedding, self.char_ids2, "char_embed2")
            char_x1_mask = tf.sequence_mask(self.char_len1, self.max_char_len1, dtype=tf.float32)
            char_x2_mask = tf.sequence_mask(self.char_len2, self.max_char_len2, dtype=tf.float32)

            word_x1_enc = tf.transpose(word_x1_enc, [1, 0, 2])
            word_x2_enc = tf.transpose(word_x2_enc, [1, 0, 2])
            word_x1_mask = tf.transpose(word_x1_mask, [1, 0])
            word_x2_mask = tf.transpose(word_x2_mask, [1, 0])

            char_x1_enc = tf.transpose(char_x1_enc, [1, 0, 2])
            char_x2_enc = tf.transpose(char_x2_enc, [1, 0, 2])
            char_x1_mask = tf.transpose(char_x1_mask, [1, 0])
            char_x2_mask = tf.transpose(char_x2_mask, [1, 0])

            word_logit = self.esim(word_x1_enc, word_x1_mask,
                                   word_x2_enc, word_x2_mask, "word", False)

            char_logit = self.esim(char_x1_enc, char_x1_mask,
                                   char_x2_enc, char_x2_mask, "char", False)

            logit = tf.concat([word_logit, char_logit], axis=-1)

            # final classifier
            with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
                logit = tf.nn.dropout(logit, (1. - self.dropout_rate))
                logit = tf.layers.dense(logit, self.fc_size,
                                        activation=tf.nn.tanh,
                                        name="fnn1",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

                logit = tf.nn.dropout(logit, (1. - self.dropout_rate))
                assert self.num_classes == 2
                logit = tf.layers.dense(logit, self.num_classes,
                                        activation=None,
                                        name="fnn2",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            self.logits = logit

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
