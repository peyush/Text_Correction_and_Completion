"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from tensorflow.contrib import learn

from utils import misc_utils as utils
from parameters import params

UNK_ID = 0


def get_table_size(vocab_file, type = 'word'): # type: char
  if(type == 'char'):
    with (open(vocab_file, 'rb')) as f:
      vocab_size = 0
      vocab_size = len(f.readlines())
  else:
    with codecs.getreader("utf-8")(open(vocab_file, 'rb')) as f:
      vocab_size = 0
      vocab_size = len(f.readlines())
  return vocab_size


def get_char_table(char_file):
  char_vocab_table = lookup_ops.index_table_from_file(char_file, default_value=UNK_ID)  
  return char_vocab_table

def check_char_vocab(vocab_file, out_dir):
  if tf.gfile.Exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    with (tf.gfile.GFile(vocab_file, 'rb')) as f:
      vocab_size = len(f.readlines())
    
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  return vocab_file, vocab_size

def check_vocab(vocab_file, out_dir, type, check_special_token=True):
  if(type == None):
    raise ValueError("Give type of language for which vocab is there")
  """Check if vocab_file doesn't exist, create from corpus_file."""
  unk = params['unk']
  sos = params['sos']
  eos = params['eos']


  if(type == 'src'):
    if tf.gfile.Exists(vocab_file):
      utils.print_out("# Vocab file %s exists" % vocab_file)
      vocab = []
      with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
          vocab_size += 1
          vocab.append(word.strip())
      if check_special_token:
        # Verify if the vocab starts with unk, sos, eos
        # If not, prepend those tokens & generate a new vocab file
        
        assert len(vocab) >= 3
        if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
          utils.print_out("The first 3 vocab words [%s, %s, %s] are not [%s, %s, %s]" %(vocab[0], vocab[1], vocab[2], unk, sos, eos))
          vocab = [unk, sos, eos] + vocab
          vocab_size += 3
          new_vocab_file = (vocab_file)
          with codecs.getwriter("utf-8")(
              tf.gfile.GFile(new_vocab_file, "wb")) as f:
            for word in vocab:
              f.write("%s\n" % word)
          vocab_file = new_vocab_file
    else:
      # Create vocab file
      new_vocab_file = (vocab_file)
      utils.print_out("# Vocab file does not :%s: given, creating one new one at :%s" % (type, new_vocab_file))
      
      with open(params['src_data_file']) as f:
        content = f.readlines()
      content = [c.strip().strip('"') for c in content]

      vocab_processor = learn.preprocessing.VocabularyProcessor(params['src_max_len'])
      vocab_processor = vocab_processor.fit(content, unused_y=None)
      vocab_dict = vocab_processor.vocabulary_._mapping
      sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
      vocab = list(list(zip(*sorted_vocab))[0])
      vocab = [unk, sos, eos] + vocab
      vocab_size = len(vocab) + 3

      with open(new_vocab_file, 'w') as f:
        for item in vocab:
          f.write("%s\n" % item)

  if(type == 'tgt'):
    if tf.gfile.Exists(vocab_file):
      utils.print_out("# Vocab file %s exists" % vocab_file)
      vocab = []
      with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
          vocab_size += 1
          vocab.append(word.strip())
      if check_special_token:
        # Verify if the vocab starts with unk, sos, eos
        # If not, prepend those tokens & generate a new vocab file
        assert len(vocab) >= 3
        if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
          utils.print_out("The first 3 vocab words [%s, %s, %s] are not [%s, %s, %s]" %(vocab[0], vocab[1], vocab[2], unk, sos, eos))
          vocab = [unk, sos, eos] + vocab
          vocab_size += 3
          new_vocab_file = (vocab_file)
          with codecs.getwriter("utf-8")(
              tf.gfile.GFile(new_vocab_file, "wb")) as f:
            for word in vocab:
              f.write("%s\n" % word)
          vocab_file = new_vocab_file
    else:
      # Create vocab file
      new_vocab_file = (vocab_file)
      utils.print_out("# Vocab file does not :%s: given, creating one new one at :%s" % (type, new_vocab_file))

      with open(params['tgt_data_file']) as f:
        content = f.readlines()
      content = [c.strip().strip('"') for c in content]

      vocab_processor = learn.preprocessing.VocabularyProcessor(params['tgt_max_len'])
      vocab_processor = vocab_processor.fit(content, unused_y=None)
      vocab_dict = vocab_processor.vocabulary_._mapping
      sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
      vocab = list(list(zip(*sorted_vocab))[0])
      vocab = [unk, sos, eos] + vocab
      with open(new_vocab_file, 'w') as f:
        for item in vocab:
          f.write("%s\n" % item)

  vocab_size = len(vocab)
  return vocab_size, vocab_file


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
  if share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table