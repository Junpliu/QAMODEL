#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2018-09-06
Author: aitingliu
"""
from __future__ import division
from __future__ import print_function

import collections
import time
import logging

import tensorflow as tf
import numpy as np

from . import vocab_utils
from . import data_helper

logger = logging.getLogger(__name__)


def print_variables_in_ckpt(ckpt_path):
    """Print a list of variables in a checkpoint together with their shapes."""
    logger.info("# Variables in ckpt %s" % ckpt_path)
    reader = tf.train.NewCheckpointReader(ckpt_path)
    variable_map = reader.get_variable_to_shape_map()
    for key in sorted(variable_map.keys()):
        logger.info("  %s: %s" % (key, variable_map[key]))


def load_model(model, ckpt, session, name):
    start_time = time.time()
    # print_variables_in_ckpt(ckpt)
    model.saver.restore(session, ckpt)
    logger.info("  loaded %s model parameters from %s, time %.2fs" %
                (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        # session.run(tf.local_variables_initializer())
        logger.info("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


class Model(collections.namedtuple("Model", ("graph", "model"))):
    pass


def create_model(model_creator, hparams, mode):
    graph = tf.Graph()
    with graph.as_default(), tf.container(mode):

        model = model_creator(hparams, mode=mode)

    return Model(
        graph=graph,
        model=model)


def _create_pretrained_emb_from_txt_const(vocab_file, embed_file, num_trainable_tokens=2,
                                          dtype=tf.float32, scope=None):
    """Load pretrain embedding from embed_file, and return an embedding matrix.
    Args:
        embed_file: Path to a Glove formated embedding txt file.
        num_trainable_tokens: Make the first n tokens in the vocab file as trainable
            variables. Default is 2, which is "<unk>" and "<pad>".
    """
    vocab, _ = vocab_utils.load_vocab(vocab_file)
    train_tokens = vocab[:num_trainable_tokens]

    logger.info("# Using pretrained embedding: %s." % embed_file)
    logger.info("  with  trainable tokens:")

    emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
    for token in train_tokens:
        logger.info("    %s" % token)
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size

    emb_mat = np.array([emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())

    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, 1])
    with tf.variable_scope(scope or "pretrain_embedding", dtype=dtype), tf.device("/cpu:0"):
        emb_mat_var = tf.get_variable(
            "emb_mat_var", [num_trainable_tokens, emb_size])
    return tf.concat([emb_mat_var, emb_mat_const], 0), emb_size


def _create_pretrained_emb_from_txt_var(vocab_file, embed_file, seed=None,
                                        dtype=tf.float32, scope=None):
    """Load pretrain embedding from embed_file, and return an embedding matrix.
    Args:
        embed_file: Path to a Glove formated embedding txt file.
    """
    np.random.seed(seed=seed)
    vocab, vocab_size = vocab_utils.load_vocab(vocab_file)

    logger.info("# Using pretrained embedding: %s." % embed_file)

    emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
    for token in vocab:
        if token not in emb_dict:
            emb_dict[token] = np.random.uniform(-0.1, 0.1, size=[emb_size])

    emb_mat = np.array([emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())

    with tf.variable_scope(scope or "pretrain_embedding", dtype=dtype):
        emb_mat_var = tf.Variable(initial_value=emb_mat, trainable=True)
    return emb_mat_var, emb_size


def create_or_load_embed(embed_name, vocab_file, embed_file, vocab_size, embed_size, dtype, trainable=True, seed=None):
    """Create a new or load an existing embedding matrix."""
    if vocab_file and embed_file:
        if trainable:
            embedding, embed_size = _create_pretrained_emb_from_txt_var(vocab_file, embed_file, seed=seed)
        else:
            embedding, embed_size = _create_pretrained_emb_from_txt_const(vocab_file, embed_file)
    else:
        unk_embed = tf.Variable(np.random.uniform(-0.1, 0.1, size=[1, embed_size]), dtype=np.float32, trainable=True)
        pad_embed = tf.Variable(np.zeros(shape=(1, embed_size)), dtype=dtype, trainable=False)
        other_embedding = tf.get_variable(
            embed_name, [vocab_size-2, embed_size], dtype,
            initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=True)
        embedding = tf.concat([unk_embed, pad_embed, other_embedding], axis=0)
    return embedding, embed_size
