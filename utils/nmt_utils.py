"""Utility functions specifically for NMT."""
from __future__ import print_function

import codecs
import time
import numpy as np
import tensorflow as tf

from utils import evaluation_utils
from utils import misc_utils as utils


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        subword_option,
                        tgt_eos,
                        num_translations_per_input=1,
                        decode=True):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  if decode:
    utils.print_out("  decoding to output %s." % trans_file)

    start_time = time.time()
    num_sentences = 0
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
      trans_f.write("")  # Write empty string to ensure file is created.

      while True:
        try:
          nmt_outputs, _ = model.decode(sess)
          if beam_width == 0:
            nmt_outputs = np.expand_dims(nmt_outputs, 0)

          batch_size = nmt_outputs.shape[1]
          num_sentences += batch_size

          for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
              translation = get_translation(
                  nmt_outputs[beam_id],
                  sent_id,
                  tgt_eos=tgt_eos,
                  subword_option=subword_option)
              trans_f.write((translation + b"\n").decode("utf-8"))
        except tf.errors.OutOfRangeError:
          utils.print_time("  done, num sentences %d, num translations per input %d" % (num_sentences, num_translations_per_input), start_time)
          break

  # Evaluation
  evaluation_scores = {}
  if ref_file and tf.gfile.Exists(trans_file):
    for metric in metrics:
      score = evaluation_utils.evaluate(
          ref_file,
          trans_file,
          metric,
          subword_option=subword_option)
      evaluation_scores[metric] = score
      utils.print_out("  %s %s: %.1f" % (metric, name, score))

  return evaluation_scores

