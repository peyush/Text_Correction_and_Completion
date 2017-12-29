
"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import numpy as np
import subprocess

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from utils import vocab_utils

def _accuracy(dev_softmax_scores = None ,targets = None, label_file = None, pred_file = None):
  """Compute accuracy, each line contains a label."""

  if((label_file!= None) and (pred_file!= None)):
    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
      with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
        count = 0.0
        match = 0.0
        for label in label_fh:
          label = label.strip()
          pred = pred_fh.readline().strip()
          if label == pred:
            match += 1
          count += 1
    accuracy =  100 * match / count
  else:
    # Per Batch 
    max_score_words = np.argmax(dev_softmax_scores, axis=2)
    count = 0.0
    match = 0.0
    for i in range(max_score_words.shape[0]):
      gt_labels = targets[i]
      pred_labels = max_score_words[i]
      if np.array_equal(gt_labels, pred_labels) == True:
        match += 1
      count += 1
    accuracy =  100 * match / count
  return accuracy
