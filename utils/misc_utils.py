"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

def get_file_row_size(file):
  with codecs.getreader("utf-8")(open(file, 'rb')) as f:
    size = 0
    for row in f:
      size += 1
  return size


def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans


def check_tensorflow_version():
  min_tf_version = "1.4.0"
  if tf.__version__ < min_tf_version:
    raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)

def add_summary(summary_writer, global_step, tag, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summary_writer.add_summary(summary, global_step)

  
def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def tokenizer(iterator):
  """
  Tokenizer generator.
  Args:
    iterator: Input iterator with strings.
  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield value.split()


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  # GPU options: https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
  config_proto = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto


def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()
