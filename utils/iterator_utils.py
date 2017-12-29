from __future__ import print_function

import collections

import tensorflow as tf


def get_iterator(src_dataset,
                 tgt_dataset,
                 char_vocab_table,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 char_pad,
                 num_buckets,
                 num_epochs,
                 src_max_len=None,
                 tgt_max_len=None,
                 src_char_max_len = None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0):

	if not output_buffer_size:
		output_buffer_size = batch_size * 1000
	src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
	tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
	tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

	char_eos_id = tf.cast(char_vocab_table.lookup(tf.constant(eos)), tf.int32)
	char_pad_id = tf.cast(char_vocab_table.lookup(tf.constant(char_pad)), tf.int32)

	src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

	src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index) # PArtition
	if skip_count is not None:
		src_tgt_dataset = src_tgt_dataset.skip(skip_count) # Skip header line

	src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size)

	src_tgt_dataset = src_tgt_dataset.map(
	  lambda src, tgt: (tf.string_split([src], delimiter="").values, tf.string_split([src]).values, tf.string_split([tgt]).values),num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

	# Filter zero length input sequences.
	src_tgt_dataset = src_tgt_dataset.filter(lambda char_src, src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

	if src_max_len:
		src_tgt_dataset = src_tgt_dataset.map(
			lambda char_src, src, tgt: (char_src, src[:src_max_len], tgt), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	if tgt_max_len:
		src_tgt_dataset = src_tgt_dataset.map(
		    lambda char_src, src, tgt: (char_src, src, tgt[:tgt_max_len]), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	
	if src_char_max_len:
		src_tgt_dataset = src_tgt_dataset.map(
		    lambda char_src, src, tgt: (char_src[:src_char_max_len], src, tgt), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	
	

	# Convert the word strings to ids.  Word strings that are not in the vocab get the lookup table's default_value integer.
	src_tgt_dataset = src_tgt_dataset.map(
	  lambda char_src, src, tgt: (tf.cast(char_vocab_table.lookup(char_src), tf.int32), tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
	  					num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	# Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
	src_tgt_dataset = src_tgt_dataset.map(
	  lambda char_src, src, tgt: (char_src, tf.concat(([tgt_sos_id], src), 0), tf.concat((tgt, [tgt_eos_id]), 0)),
	  					num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	# Add in sequence lengths.
	src_tgt_dataset = src_tgt_dataset.map(
	  lambda char_src, src, tgt: ( char_src, src, tgt, tf.size(char_src),tf.size(src), tf.size(tgt)),
	  					num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	# Readt after epoch
	src_tgt_dataset = src_tgt_dataset.repeat()


	# Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
	def batching_func(x):
		return x.padded_batch(batch_size,
			    # The first three entries are the source and target line rows;
			    # these have unknown-length vectors.  The last two entries are
			    # the source and target row sizes; these are scalars.
			    padded_shapes=(
			        tf.TensorShape([src_char_max_len]),  # char_src
			        tf.TensorShape([None]),  # src_input
			        tf.TensorShape([None]),  # tgt_output
			        tf.TensorShape([]), # char_len
			        tf.TensorShape([]),  # src_len
			        tf.TensorShape([])),  # tgt_len
			    # Pad the source and target sequences with eos tokens.
			    # (Though notice we don't generally need to do this since
			    # later on we will be masking out calculations past the true sequence.
			    padding_values=(
			    	char_pad_id,
			        src_eos_id,  # src
			        tgt_eos_id,  # tgt_input
			        0, # char_len -- unused
			        0,  # src_len -- unused
			        0))  # tgt_len -- unused

	if num_buckets > 1:

		def key_func(unused_1, unused_2, unused_3,char_len,  src_len, tgt_len):
			# Calculate bucket_width by maximum source sequence length.
			# Pairs with length [0, bucket_width) go to bucket 0, length
			# [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
			# over ((num_bucket-1) * bucket_width) words all go into the last bucket.
			if src_max_len:
				bucket_width = (src_max_len + num_buckets - 1) // num_buckets
			else:
				bucket_width = 10

			# Bucket sentence pairs by the length of their source sentence and target sentence.
			bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
			return tf.to_int64(tf.minimum(num_buckets, bucket_id))

		def reduce_func(unused_key, windowed_data):
			return batching_func(windowed_data)

		batched_dataset = src_tgt_dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
	else:
		batched_dataset = batching_func(src_tgt_dataset)

	batched_iter = batched_dataset.make_initializable_iterator()
	#(char_src_ids, src_input_ids, tgt_output_ids, char_seq_len, src_seq_len, tgt_seq_len) = (batched_iter.get_next())
	return batched_iter