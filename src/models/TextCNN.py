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

logger = logging.getLogger(__name__)


class TextCNN(object):
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
        self.learning_rate = tf.constant(config.learning_rate)
        self.opt = config.opt
        self.max_gradient_norm = config.max_gradient_norm
        self.num_keep_ckpts = config.num_keep_ckpts
        self.batch_size = config.batch_size

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

    def textcnn(self, inputs, max_seq_len, embedding_size, filter_sizes, num_filters, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            embeded_seq_expanded = tf.expand_dims(inputs, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # convolution layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable("W", shape=filter_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b", shape=[num_filters], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(
                        embeded_seq_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, max_seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            h_pool_flat = tf.nn.dropout(h_pool_flat, 1.0 - self.dropout)

        return h_pool_flat

    def build_graph(self):
        with tf.variable_scope("net"):
            self.init_embedding()
            word_embed1 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids1, "word_embed1")
            word_embed2 = tf.nn.embedding_lookup(self.word_embedding, self.word_ids2, "word_embed2")
            char_embed1 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids1, "char_embed1")
            char_embed2 = tf.nn.embedding_lookup(self.char_embedding, self.char_ids2, "char_embed2")

            word_rep1 = self.textcnn(word_embed1, self.max_word_len1, self.word_embed_size,
                                     self.filter_sizes, self.num_filters, tf.AUTO_REUSE, "word")
            word_rep2 = self.textcnn(word_embed2, self.max_word_len2, self.word_embed_size,
                                     self.filter_sizes, self.num_filters, tf.AUTO_REUSE, "word")

            char_rep1 = self.textcnn(char_embed1, self.max_char_len1, self.char_embed_size,
                                     self.filter_sizes, self.num_filters, tf.AUTO_REUSE, "char")
            char_rep2 = self.textcnn(char_embed2, self.max_char_len2, self.char_embed_size,
                                     self.filter_sizes, self.num_filters, tf.AUTO_REUSE, "char")

            sent_merge1 = tf.concat([word_rep1, char_rep1], axis=1, name="sent_merge1")
            sent_merge2 = tf.concat([word_rep2, char_rep2], axis=1, name="sent_merge2")

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
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
                losses = tf.reduce_mean(losses)
                self.l2_losses = tf.add_n(
                    [tf.nn.l2_loss(tf.cast(v, self.dtype)) for v in tf.trainable_variables()],
                    name="l2_losses") * self.l2_reg_lambda
                self.losses = tf.add(losses, self.l2_losses, name="loss")
                tf.summary.scalar("loss", self.losses)

            with tf.variable_scope("metrics"):
                self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.pred_labels, name="acc")
                self.recall, self.recall_op = tf.metrics.recall(self.labels, self.pred_labels, name="rec")
                self.precision, self.precision_op = tf.metrics.precision(self.labels, self.pred_labels, name="pre")
                self.auc, self.auc_op = tf.metrics.auc(self.labels, self.pred_labels, name="auc")

                tf.summary.scalar("accuracy", self.accuracy)
                tf.summary.scalar("recall", self.recall)
                tf.summary.scalar("precision", self.precision)
                tf.summary.scalar("auc", self.auc)

    def add_train_op(self):
        params = tf.trainable_variables()
        with tf.variable_scope("opt"):
            # self.learning_rate = tf.train.exponential_decay(
            #     learning_rate=self.learning_rate,
            #     global_step=self.global_step,
            #     decay_steps=1000,
            #     decay_rate=0.96,
            #     staircase=True,
            #     name="learning_rate_decay")
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
            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            self.grad_norm = grad_norm
            tf.summary.scalar("learning_rate", self.learning_rate)
            tf.summary.scalar("grad_norm", self.grad_norm)
            tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))

    def build(self):
        self.add_placeholders()
        self.build_graph()
        self.add_loss_op()

        if self.mode == "train":
            self.add_train_op()

        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_keep_ckpts)

        params = tf.trainable_variables()
        logger.info("# Training variables")
        for param in params:
            logging.info("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

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

        output_feed = [self.simscore, self.losses,
                       self.train_op, self.accuracy_op, self.recall_op, self.precision_op, self.auc_op,
                       self.global_step, self.grad_norm, self.learning_rate, self.simscore]
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
        output_feed = [self.simscore, self.losses,
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
