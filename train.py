"""For training NMT models."""
from __future__ import print_function

import numpy as np
import math
import os
import codecs
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


from utils import misc_utils as utils
from utils import vocab_utils
from utils import iterator_utils
from utils import evaluation_utils
from tensorflow.python.ops import lookup_ops
from parameters import params

import model as nmt_model


utils.check_tensorflow_version()



def train():
	"""Train a translation model."""
	create_new_model = params['create_new_model']
	out_dir = params['out_dir']
	model_creator = nmt_model.Model # Create model graph
	summary_name = "train_log"

	# Setting up session and initilize input data iterators
	src_file = params['src_data_file']
	tgt_file = params['tgt_data_file']
	dev_src_file = params['dev_src_file']
	dev_tgt_file = params['dev_tgt_file']
	test_src_file = params['test_src_file']
	test_tgt_file = params['test_tgt_file']


	char_vocab_file = params['enc_char_map_path']
	src_vocab_file = params['src_vocab_file']
	tgt_vocab_file = params['tgt_vocab_file']
	if(src_vocab_file == '' or src_vocab_file == ''):
		raise ValueError("vocab_file '%s' not given in params.") 

	graph = tf.Graph()

  	# Log and output files
	log_file = os.path.join(out_dir, "log_%d" % time.time())
	log_f = tf.gfile.GFile(log_file, mode="a")
	utils.print_out("# log_file=%s" % log_file, log_f)


	# Model run params
	num_epochs = params['num_epochs']
	batch_size = params['batch_size']
	steps_per_stats = params['steps_per_stats']

	utils.print_out("# Epochs=%s, Batch Size=%s, Steps_per_Stats=%s" % (num_epochs, batch_size, steps_per_stats), None)

	with graph.as_default():
		src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(src_vocab_file, tgt_vocab_file, params['share_vocab'])
		char_vocab_table = vocab_utils.get_char_table(char_vocab_file)
		reverse_target_table = lookup_ops.index_to_string_table_from_file(tgt_vocab_file, default_value=params['unk'])

		src_dataset = tf.data.TextLineDataset(src_file)
		tgt_dataset = tf.data.TextLineDataset(tgt_file)

		batched_iter = iterator_utils.get_iterator(src_dataset,
											   tgt_dataset,
											   char_vocab_table,
											   src_vocab_table,
											   tgt_vocab_table,
											   batch_size=batch_size,
											   sos=params['sos'],
											   eos=params['eos'],
											   char_pad = params['char_pad'],
											   num_buckets=params['num_buckets'],
											   num_epochs = params['num_epochs'],
											   src_max_len=params['src_max_len'],
											   tgt_max_len=params['tgt_max_len'],
											   src_char_max_len = params['char_max_len']
											   )

		# Summary writer
		summary_writer = tf.summary.FileWriter(os.path.join(out_dir, summary_name),graph)


		# Preload validation data for decoding.
		dev_src_dataset = tf.data.TextLineDataset(dev_src_file)
		dev_tgt_dataset = tf.data.TextLineDataset(dev_tgt_file)
		dev_batched_iterator = iterator_utils.get_iterator(dev_src_dataset,
														   dev_tgt_dataset,
														   char_vocab_table,
														   src_vocab_table,
														   tgt_vocab_table,
														   batch_size=batch_size,
														   sos=params['sos'],
														   eos=params['eos'],
														   char_pad = params['char_pad'],
														   num_buckets=params['num_buckets'],
														   num_epochs = params['num_epochs'],
														   src_max_len=params['src_max_len'],
														   tgt_max_len=params['tgt_max_len'],
														   src_char_max_len = params['char_max_len']
														   )

		# Preload test data for decoding.
		test_src_dataset = tf.data.TextLineDataset(test_src_file)
		test_tgt_dataset = tf.data.TextLineDataset(test_tgt_file)
		test_batched_iterator = iterator_utils.get_iterator(test_src_dataset,
														   test_tgt_dataset,
														   char_vocab_table,
														   src_vocab_table,
														   tgt_vocab_table,
														   batch_size=batch_size,
														   sos=params['sos'],
														   eos=params['eos'],
														   char_pad = params['char_pad'],
														   num_buckets=params['num_buckets'],
														   num_epochs = params['num_epochs'],
														   src_max_len=params['src_max_len'],
														   tgt_max_len=params['tgt_max_len'],
														   src_char_max_len = params['char_max_len']
														   )

		config_proto = utils.get_config_proto(log_device_placement=params['log_device_placement'])
		sess = tf.Session(config=config_proto)


		with sess.as_default():
			

			train_model = model_creator(mode = params['mode'],
										train_iterator = batched_iter,
										val_iterator = dev_batched_iterator,
										char_vocab_table = char_vocab_table,
										source_vocab_table=src_vocab_table,
										target_vocab_table=tgt_vocab_table,
										reverse_target_table = reverse_target_table)

			loaded_train_model, global_step = create_or_load_model(train_model, params['out_dir'],session=sess,name="train",
																	log_f = log_f, create=create_new_model)
			
			sess.run([batched_iter.initializer,dev_batched_iterator.initializer, test_batched_iterator.initializer])


			start_train_time = time.time()
			utils.print_out("# Start step %d, lr %g, %s" %(global_step, loaded_train_model.learning_rate.eval(session=sess), time.ctime()), log_f)
			
			# Reset statistics
			stats = init_stats()

			steps_per_epoch = int(np.ceil(utils.get_file_row_size(src_file) / batch_size))
			utils.print_out("Total steps per epoch: %d" % steps_per_epoch)

			def train_step(model, sess):	
				return model.train(sess)
			def dev_step(model, sess):
				total_steps = int(np.ceil(utils.get_file_row_size(dev_src_file) / batch_size ))
				total_dev_loss = 0.0
				total_accuracy = 0.0
				for _ in range(total_steps):
					dev_result_step = model.dev(sess)
					dev_softmax_scores, dev_loss, tgt_output_ids,_,_,_,_ = dev_result_step
					total_dev_loss += dev_loss * params['batch_size']
					total_accuracy += evaluation_utils._accuracy(dev_softmax_scores, tgt_output_ids,  None, None)
				return (total_dev_loss/total_steps, total_accuracy/total_steps)


			for epoch_step in range(num_epochs): 
				for curr_step in range(int(np.ceil(steps_per_epoch))):
					start_time = time.time()
					step_result = train_step(loaded_train_model, sess)
					global_step = update_stats(stats, summary_writer, start_time, step_result)

    				# Logging Step
					if(curr_step % params['steps_per_stats'] == 0):
						check_stats(stats, global_step, steps_per_stats, log_f)



					# Evaluation
					if(curr_step % params['steps_per_devRun'] == 0):
						dev_step_loss, dev_step_acc = dev_step(loaded_train_model, sess)
						utils.print_out("Dev Step total loss, Accuracy: %f, %f" % (dev_step_loss, dev_step_acc), log_f)

				utils.print_out("# Finished an epoch, epoch completed %d" % epoch_step)
				loaded_train_model.saver.save(sess,  os.path.join(out_dir, "translate.ckpt"), global_step=global_step)
				dev_step_loss = dev_step(loaded_train_model, sess)


			utils.print_time("# Done training!", start_train_time)
			summary_writer.close()


def update_stats(stats, summary_writer, start_time, step_result):
	"""Update stats: write summary and accumulate statistics."""
	(debug , _, step_loss, step_predict_count, step_summary, global_step, step_word_count) = step_result
	#print (debug.shape)

	# Write step summary.
	summary_writer.add_summary(step_summary, global_step)

	# update statistics
	stats["step_time"] += (time.time() - start_time)
	stats["loss"] = (step_loss * params['batch_size'])
	stats["predict_count"] += step_predict_count
	stats["total_count"] += float(step_word_count)

	return global_step


def check_stats(stats, global_step, steps_per_stats, log_f):
	"""Print statistics."""
	# Print statistics for the previous epoch.
	avg_step_time = stats["step_time"] / steps_per_stats
	speed = stats["total_count"] / (1000 * stats["step_time"])
	

	utils.print_out("  global step %d loss %.3f step-time %.2fs wps %.2fK" %
		(global_step, stats["loss"],avg_step_time, speed), log_f)

def init_stats():
	"""Initialize statistics that we want to keep."""
	return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0, "total_count": 0.0}

def create_or_load_model(model, model_dir, session, name, log_f,create = True):

	"""Create translation model and initialize or load parameters in session."""

	if create == False:
		latest_ckpt = tf.train.latest_checkpoint(model_dir)
		model = load_model(model, latest_ckpt, session, name)
	else:
		start_time = time.time()
		session.run(tf.global_variables_initializer())
		session.run(tf.tables_initializer())
		utils.print_out("  created %s model with fresh parameters, time %.2fs" %     (name, time.time() - start_time), log_f)

	global_step = model.global_step.eval(session=session)
	return model, global_step


def load_model(model, ckpt, session, name):
	start_time = time.time()
	model.saver.restore(session, ckpt)
	session.run(tf.tables_initializer())
	utils.print_out("  loaded %s model parameters from %s, time %.2fs" %(name, ckpt, time.time() - start_time))
	return model



def load_data(inference_input_file):
	"""Load inference data."""
	with codecs.getreader("utf-8")(tf.gfile.GFile(inference_input_file, mode="rb")) as f:
		inference_data = f.read().splitlines()
	return inference_data


def run_main(unused_argv):
	"""Run main."""

	# Initialization, Vocab generation
	if not tf.gfile.Exists(params['out_dir']):
		utils.print_out("# Creating output directory %s ..." % params['out_dir'])
		tf.gfile.MakeDirs(params['out_dir'])

	char_vocab_file = params['enc_char_map_path']
	src_vocab_file = params['src_vocab_file']
	tgt_vocab_file = params['tgt_vocab_file']


	char_vocab_size, char_vocab_file = vocab_utils.check_char_vocab(char_vocab_file, params['out_dir'])

	src_vocab_size, src_vocab_file = vocab_utils.check_vocab(src_vocab_file, params['out_dir'], type = 'src',
	                                                        check_special_token=params['check_special_token'])
	tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(tgt_vocab_file,
                                                            params['out_dir'], type = 'tgt',
                                                            check_special_token=params['check_special_token'])

	## Train / Decode
	if params['mode'] == 'infer':
	# # Modification required#########
	# # Inference
	# trans_file = params['inference_output_file']
	# ckpt = params['ckpt']
	# if not ckpt:
	#   ckpt = tf.train.latest_checkpoint(out_dir)
	# inference_fn(ckpt, inference_input_file, trans_file, num_workers, jobid)

	# # Evaluation
	# ref_file = params['inference_ref_file']
	# if ref_file and tf.gfile.Exists(trans_file):
	#   for metric in params['metrics']:
	#     score = evaluation_utils.evaluate(ref_file,
				# 						  trans_file,
				# 						  metric,
				# 						  params['subword_option'])
	#     utils.print_out("  %s: %.1f" % (metric, score))
		pass
		infer()
	elif(params['mode'] == 'train'):
		# Train
		train()



if __name__ == "__main__":
	tf.app.run(main=run_main, argv=None)
