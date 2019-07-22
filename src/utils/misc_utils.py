#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import codecs
import json
from functools import reduce

import tensorflow as tf

logger = logging.getLogger(__name__)


def get_config_proto(log_device_placement=False, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config_proto = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    #
    # config_proto = tf.ConfigProto(
    #     log_device_placement=log_device_placement,
    #     allow_soft_placement=allow_soft_placement)
    # config_proto.gpu_options.allow_growth = True

    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto


def print_params():
    def size(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list())

    params = tf.trainable_variables()
    logger.info("# Training variables")
    for param in params:
        logging.info("  %s, %s, %s" % (param.name, str(param.get_shape()), param.device))
    logging.info("# Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))


def print_hparams(hparams, skip_patterns=None, header=None):
    """Print hparams, can skip keys based on pattern."""
    if header:
        logger.info("%s" % header)
    values = hparams.values()
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key for skip_pattern in skip_patterns]):
            logger.info("  %s=%s" % (key, str(values[key])))


def load_hparams(hparams_file):
    """Load hparams from a hparams_file."""

    logger.info("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
        try:
            hparams_values = json.load(f)
            hparams = tf.contrib.training.HParams(**hparams_values)
        except ValueError:
            logger.info("  can't load hparams file")
            return None
    return hparams


def save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    logger.info("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json(indent=4, sort_keys=True))
