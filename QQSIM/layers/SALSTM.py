#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2019-08-16
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


class SALSTM(object):
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
        self.unit_type = config.unit_type
        self.num_units = config.num_units
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
        self.d_a_size = config.d_a_size
        self.r_size = config.r_size

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

    def bi_rnn(self, inputs, sequence_length, num_units, unit_type, dtype, reuse, name):
        """Create and call bidirectional RNN cells.
        Returns:
            The concatenated bidirectional output and the bidirectional RNN cell's state.
        """
        with tf.variable_scope(name, reuse=reuse):
            if self.unit_type == "lstm":
                fw_cell = tf.contrib.rnn.LSTMCell(num_units)
                bw_cell = tf.contrib.rnn.LSTMCell(num_units)
            elif self.unit_type == "gru":
                fw_cell = tf.contrib.rnn.GRUCell(num_units)
                bw_cell = tf.contrib.rnn.GRUCell(num_units)
            else:
                raise ValueError("Unknown unit type %s!" % unit_type)

            # TODO: whether use dropout
            if self.mode == "train":
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=fw_cell,
                                                        output_keep_prob=(1.0 - self.dropout))
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=bw_cell,
                                                        output_keep_prob=(1.0 - self.dropout))

            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                inputs,
                sequence_length=sequence_length,
                dtype=dtype)
            state_fw, state_bw = bi_state
            bi_state_h = tf.concat([state_fw.h, state_bw.h], -1)

        return tf.concat(bi_outputs, -1), bi_state_h

    def self_attention(self, inputs, max_seq_len, hidden_size, d_a_size, r_size, reuse, name):
        """
        This implements the self attention from the paper:

          https://arxiv.org/abs/1703.03130

        Lin Z, Feng M, Santos C N, et al. A structured self-attentive sentence embedding[J].
        arXiv preprint arXiv:1703.03130, 2017.

        Args:
          inputs: input `Tensor`, shaped `[batch_size, max_seq_len, 2 * hidden_size]`.
          max_seq_len: max sequence length
          hidden_size: lstm hidden size
          d_a_size: set same as paper
          r_size: set same as paper
          reuse:
          name: string; Create variable names under this scope.

        Returns:
          output `Tensor`, shaped `[batch_size, 2 * hidden_size * r_size]`.
        """
        with tf.variable_scope(name+"/self_attention", reuse=reuse):
            H = tf.reshape(inputs, [-1, 2 * hidden_size])  # [N*L, 2*H]
            W_s1 = tf.get_variable("W_s1", shape=[2 * hidden_size, d_a_size])  # [2*H, da]
            H_s1 = tf.tanh(tf.matmul(H, W_s1))  # [N*L, da]
            W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size])  # [da, r]
            H_s2 = tf.matmul(H_s1, W_s2)  # [N*L, r]
            H_s2_reshape = tf.transpose(
                tf.reshape(H_s2, [-1, max_seq_len, r_size]),
                [0, 2, 1])  # [N, r, L]
            A = tf.nn.softmax(H_s2_reshape, name="attention")
            M = tf.matmul(A, inputs)  # [N, r, 2*H]
            M_flat = tf.reshape(M, shape=[-1, 2 * hidden_size * r_size])  # [N, 2*H*r]

            # A_T = tf.transpose(A, perm=[0, 2, 1])
            # tile_eye = tf.tile(tf.eye(r_size), [self.batch_size, 1])
            # tile_eye = tf.reshape(tile_eye, [-1, r_size, r_size])
            # AA_T = tf.matmul(A, A_T) - tile_eye
            # self.P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

            outputs = M_flat
            return outputs

    def build_graph(self):
        with tf.variable_scope("net"):
            self.init_embedding()
            word_embed1 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids1, "word_embed1")
            word_embed2 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids2, "word_embed2")
            char_embed1 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids1, "char_embed1")
            char_embed2 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids2, "char_embed2")

            word_out1, word_rep1 = self.bi_rnn(word_embed1, self.word_len1, self.num_units,
                                               self.unit_type, self.dtype, tf.AUTO_REUSE, "word")
            word_out2, word_rep2 = self.bi_rnn(word_embed2, self.word_len2, self.num_units,
                                               self.unit_type, self.dtype, tf.AUTO_REUSE, "word")

            word_att1 = self.self_attention(word_out1, self.max_word_len1, self.num_units, self.d_a_size, self.r_size, tf.AUTO_REUSE, "word")
            word_att2 = self.self_attention(word_out2, self.max_word_len2, self.num_units, self.d_a_size, self.r_size, tf.AUTO_REUSE, "word")

            char_out1, char_rep1 = self.bi_rnn(char_embed1, self.char_len1, self.num_units,
                                               self.unit_type, self.dtype, tf.AUTO_REUSE, "char")
            char_out2, char_rep2 = self.bi_rnn(char_embed2, self.char_len2, self.num_units,
                                               self.unit_type, self.dtype, tf.AUTO_REUSE, "char")
            
            char_att1 = self.self_attention(char_out1, self.max_char_len1, self.num_units, self.d_a_size, self.r_size, tf.AUTO_REUSE, "char")
            char_att2 = self.self_attention(char_out2, self.max_char_len2, self.num_units, self.d_a_size, self.r_size, tf.AUTO_REUSE, "char")

            sent_merge1 = tf.concat([word_att1, char_att1], axis=1, name="sent_merge1")
            sent_merge2 = tf.concat([word_att2, char_att2], axis=1, name="sent_merge2")

            reps_cat = tf.concat([sent_merge1, sent_merge2], axis=1)
            reps_add = tf.add(sent_merge1, sent_merge2)
            reps_sub = tf.subtract(sent_merge1, sent_merge2)
            reps_abs_sub = tf.abs(tf.subtract(sent_merge1, sent_merge2))
            reps_mul = tf.multiply(sent_merge1, sent_merge2)
            reps_match = tf.concat([reps_cat, reps_add, reps_sub, reps_abs_sub, reps_mul], axis=1)

            sent_dense = tf.layers.dense(inputs=reps_match, units=self.fc_size, activation=tf.nn.relu)
            sent_dense = tf.nn.dropout(sent_dense, keep_prob=(1.0 - self.dropout))
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
