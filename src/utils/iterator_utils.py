# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from . import vocab_utils

COLUMNS_TRAIN = ["idx", "seq1", "seq2", "label"]
COLUMNS_INFER = ["idx", "seq1", "seq2"]

FIELD_DEFAULT_TRAIN = [[""], [""], [""], [0]]
FIELD_DEFAULT_INFER = [[""], [""], [""]]


def _parse_line_train(line):
    fields = tf.decode_csv(line, FIELD_DEFAULT_TRAIN, field_delim="\t", use_quote_delim=False)
    columns = dict(zip(COLUMNS_TRAIN, fields))
    idx = columns.pop("idx")
    seq1 = columns.pop("seq1")
    seq2 = columns.pop("seq2")
    label = columns.pop("label")
    return idx, seq1, seq2, label


def _parse_line_test(line):
    fields = tf.decode_csv(line, FIELD_DEFAULT_INFER, field_delim="\t", use_quote_delim=False)
    columns = dict(zip(COLUMNS_INFER, fields))
    idx = columns.pop("idx")
    seq1 = columns.pop("seq1")
    seq2 = columns.pop("seq2")
    return idx, seq1, seq2


class BatchInput(
    collections.namedtuple("BatchTrainInput",
                           ("initializer",
                            "label", "idx",
                            "seq1", "seq1_length",
                            "seq2", "seq2_length"))):
    pass


def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 random_seed=None,
                 seq1_max_len=None,
                 seq2_max_len=None,
                 reshuffle_each_iteration=True,
                 mode="train"):
    dataset = dataset.map(_parse_line_train)

    if mode == "train":
        dataset = dataset.shuffle(
            10000, random_seed, reshuffle_each_iteration)

    dataset = dataset.map(
        lambda idx, seq1, seq2, label: (
            idx, tf.string_split([seq1]).values, tf.string_split([seq2]).values, label))

    if mode == "train":
        # Filter zero length input sequences.
        dataset = dataset.filter(
            lambda idx, seq1, seq2, label: tf.logical_and(tf.size(seq1) > 0, tf.size(seq2) > 0))

    if seq1_max_len:
        dataset = dataset.map(
            lambda idx, seq1, seq2, label: (idx, seq1[:seq1_max_len], seq2, label))

    if seq2_max_len:
        dataset = dataset.map(
            lambda idx, seq1, seq2, label: (idx, seq1, seq2[:seq2_max_len], label))

    # Convert the word strings to ids. Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(
        lambda idx, seq1, seq2, label: (
            idx,
            tf.cast(vocab_table.lookup(seq1), tf.int32),
            tf.cast(vocab_table.lookup(seq2), tf.int32),
            label))

    # Add in sequence lengths.
    dataset = dataset.map(
        lambda idx, seq1, seq2, label: (idx, seq1, seq2, label, tf.size(seq1), tf.size(seq2)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape([seq1_max_len]),
                tf.TensorShape([seq2_max_len]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([])),
            padding_values=(
                "",
                vocab_utils.PAD_ID,
                vocab_utils.PAD_ID,
                0,  # unused
                0,  # unused
                0)  # unused
        )

    batched_dataset = batching_func(dataset)

    batch_iter = batched_dataset.make_initializable_iterator()
    (idx, seq1_ids, seq2_ids, label, seq1_len, seq2_len) = batch_iter.get_next()

    return BatchInput(
        initializer=batch_iter.initializer,
        idx=idx,
        seq1=seq1_ids,
        seq2=seq2_ids,
        label=label,
        seq1_length=seq1_len,
        seq2_length=seq2_len)


def get_infer_iterator(dataset,
                       vocab_table,
                       batch_size,
                       seq1_max_len=None,
                       seq2_max_len=None):
    dataset = dataset.map(_parse_line_test)

    dataset = dataset.map(
        lambda idx, seq1, seq2: (idx, tf.string_split([seq1]).values, tf.string_split([seq2]).values))

    if seq1_max_len:
        dataset = dataset.map(
            lambda idx, seq1, seq2: (idx, seq1[:seq1_max_len], seq2))

    if seq2_max_len:
        dataset = dataset.map(
            lambda idx, seq1, seq2: (idx, seq1, seq2[:seq2_max_len]))

    dataset = dataset.map(
        lambda idx, seq1, seq2: (
            idx,
            tf.cast(vocab_table.lookup(seq1), tf.int32),
            tf.cast(vocab_table.lookup(seq2), tf.int32)))

    dataset = dataset.map(
        lambda idx, seq1, seq2: (
            idx, seq1, seq2, tf.size(seq1), tf.size(seq2)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape([seq1_max_len]),
                tf.TensorShape([seq2_max_len]),
                tf.TensorShape([]),
                tf.TensorShape([])),
            padding_values=(
                "",
                vocab_utils.PAD_ID,
                vocab_utils.PAD_ID,
                0,
                0))

    batched_dataset = batching_func(dataset)
    batch_iter = batched_dataset.make_initializable_iterator()
    (idx, seq1_ids, seq2_ids, seq1_len, seq2_len) = batch_iter.get_next()

    return BatchInput(
        initializer=batch_iter.initializer,
        idx=idx,
        seq1=seq1_ids,
        seq2=seq2_ids,
        seq1_length=seq1_len,
        seq2_length=seq2_len,
        label=None)
