import os
import sys
import json
import time
import datetime
import logging

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from parameters import params

from utils import misc_utils as utils
from utils import vocab_utils

def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm


class Model(object):
    def __init__(self, mode, train_iterator, val_iterator, char_vocab_table,
                 source_vocab_table, target_vocab_table, reverse_target_table = None, scope=None):

        self.mode = mode
        self.debug = None

        self.iter_type = tf.placeholder(tf.int32, name="iterator_type") # 1- train, 2 - val
        self.set_data_placeholders(train_iterator, val_iterator)


        self.char_vocab_size = vocab_utils.get_table_size(params['enc_char_map_path'], 'char')
        self.src_vocab_size = vocab_utils.get_table_size(params['src_vocab_file'], 'word')
        self.tgt_vocab_size = vocab_utils.get_table_size(params['tgt_vocab_file'], 'word')

        self.char_vocab_table = char_vocab_table
        self.source_vocab_table = source_vocab_table
        self.target_vocab_table = target_vocab_table

        self.time_major = params['time_major']

        self.batch_size = params['batch_size']
		# Initializer
        initializer = get_initializer(init_op=params['init_op'],seed = None,init_weight =params['init_weight'])
        tf.get_variable_scope().set_initializer(initializer)

		# Embeddings
		# TODO(ebrevdo): Only do this if the mode is TRAIN?
        self.init_embeddings(scope)

		# Projection
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(self.tgt_vocab_size, use_bias=False, name="output_projection")

        ## Train graph
        res = self.build_graph(scope=scope)

        if self.mode == 'train':
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(self.source_sequence_length) + tf.reduce_sum(self.target_sequence_length)
        self.logits = res[0]

        self.target_words = reverse_target_table.lookup(tf.to_int64(self.tgt_output_ids))

        # else:
        #     self.infer_logits, _, self.final_context_state, self.sample_id = res
        #     self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))

        if self.mode != 'infer':
            ## Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(self.target_sequence_length)

        self.global_step = tf.Variable(0, trainable=False)
        model_params = tf.trainable_variables()

	    # Gradients and SGD update operation for training the model.
	    # Arrage for the embedding vars to appear at the beginning.
        if self.mode == 'train':
            self.learning_rate = tf.constant(params['learning_rate'])

			# Optimizer
            if params['optimizer'] == "sgd":
            	opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            	tf.summary.scalar("lr", self.learning_rate)
            elif params['optimizer'] == "adam":
            	opt = tf.train.AdamOptimizer(self.learning_rate)

			# Gradients
            gradients = tf.gradients(self.train_loss, model_params, colocate_gradients_with_ops=params['colocate_gradients_with_ops'])

            clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients, max_gradient_norm=params['max_gradient_norm'])
            self.grad_norm = grad_norm

            #self.update = opt.apply_gradients(zip(clipped_grads, model_params), global_step=self.global_step)
            self.update = opt.minimize(self.train_loss)

            self.softmax_scores = tf.nn.softmax(self.logits, dim=-1, name='softmax_Sequence')

            # Summary
            self.train_summary = tf.summary.merge([tf.summary.scalar("lr", self.learning_rate),
												tf.summary.scalar("train_loss", self.train_loss),] + grad_norm_summary)

        self.debug = self.logits
	    # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        utils.print_out("# Trainable variables")
        for parameter in model_params:
            utils.print_out("  %s, %s, %s" % (parameter.name, str(parameter.get_shape()), parameter.op.device))


    def init_embeddings(self, scope):
        """Init embeddings."""
        self.embedding_char_encoder, self.embedding_src_decoder, _ = self.create_emb_for_encoder_and_decoder(char_vocab_size = self.char_vocab_size,
                                                                                                        src_vocab_size=self.src_vocab_size,
                                                                                                        tgt_vocab_size=self.tgt_vocab_size,
                                                                                                        src_embed_size=params['num_units'],
                                                                                                        tgt_embed_size=params['num_units'],
                                                                                                        scope=scope)


    def create_emb_for_encoder_and_decoder(self, char_vocab_size, src_vocab_size, tgt_vocab_size, src_embed_size, tgt_embed_size, dtype=tf.float32, scope=None):
        """
        Create embedding matrix for both encoder and decoder.
            Args:
                char_vocab_size
                src_vocab_size: An integer. The source vocab size.
                tgt_vocab_size: An integer. The target vocab size.
                src_embed_size: An integer. The embedding dimension for the encoder's embedding.
                tgt_embed_size: An integer. The embedding dimension for the decoder's embedding.
                dtype: dtype of the embedding matrix. Default to float32.
                scope: VariableScope for the created subgraph. Default to "embedding".
            Returns:
            embedding_encoder: Encoder's embedding matrix.
            embedding_decoder: Decoder's embedding matrix.
        """
        
        with tf.variable_scope(scope or "embeddings", dtype=dtype) as scope:
            with tf.variable_scope("char_encoder"):
                embedding_char_encoder = tf.get_variable("embedding_char_encoder", [char_vocab_size, src_embed_size], dtype)

            with tf.variable_scope("src_decoder"):
                embedding_src_decoder = tf.get_variable("embedding_src_decoder", [src_vocab_size, tgt_embed_size], dtype)

            with tf.variable_scope("tgt_decoder"):
                embedding_tgt_decoder = tf.get_variable("embedding_tgt_decoder", [tgt_vocab_size, tgt_embed_size], dtype)

        return embedding_char_encoder, embedding_src_decoder, embedding_tgt_decoder

    

    def build_graph(self, scope=None):
        '''
        Subclass must implement this method.
        Creates a sequence-to-sequence model with dynamic RNN decoder API.
        Args:
            scope: VariableScope for the created subgraph; default "dynamic_seq2seq".
        Returns:
            A tuple of the form (logits, loss, final_context_state),
            where:
            logits: float32 Tensor [batch_size x num_decoder_symbols].
            loss: the total loss / batch_size.
            final_context_state: The final state of decoder RNN.
        Raises:
            ValueError: if encoder_type differs from mono and bi, or
            attention_option is not (luong | scaled_luong |  bahdanau | normed_bahdanau).

        '''
        utils.print_out("# creating %s graph ..." % self.mode)
        dtype = tf.float32
        num_layers = params['num_layers']

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            # Encoder
            encoder_outputs, encoder_state = self._build_encoder()

            ## Decoder
            logits, sample_id, final_context_state = self._build_decoder(encoder_outputs, encoder_state)

            ## Loss
            if self.mode == 'train':
                loss = self._compute_loss(logits)
            else:
                loss = None

        return logits, loss, final_context_state


    def _build_encoder(self):
        """Build an encoder."""
        num_layers = params['num_layers']

        source = self.char_src_ids
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units] Time_major: True
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_char_encoder, source)
            
           

            ##### Character RNN ###########
            # Encoder_outpus: [max_time, batch_size, num_units],Time_major: True            
            utils.print_out("RNN Encoder num_layers = %d "%(num_layers))
            rnn_cell = self._build_encoder_rnn_cell(params, num_layers)
            # State: [Batch_size * num_units]
            encoder_output_1, encoder_state_1 = tf.nn.dynamic_rnn(rnn_cell,encoder_emb_inp,
                                                                dtype=dtype,
                                                                sequence_length = self.char_sequence_length,
                                                                time_major=self.time_major,
                                                                swap_memory=True)

           

            ##### Character CNN ###########
            utils.print_out("Building encoder CNN layers")
            # Output: [Batch_size * num_units]
            if(self.time_major):
                conv_input = tf.transpose(encoder_emb_inp, [1,0,2])
            else:
                conv_input = encoder_emb_inp


            encoder_output_2 = self.cnn_encoder_output( conv_input, 
                                                        mode = self.mode,
                                                        output_shape = params['num_units'],
                                                        num_filters = params['num_filters_cnn'],
                                                        filter_sizes = params['filter_sizes'],
                                                        embedding_size = params['num_units'],
                                                        sequence_length = params['char_max_len'],
                                                        dropout = params['dropout'],)
            ###### Merging to get decoder state###########
            encoder_outputs, encoder_state = self.encode_Combine(encoder_output_1, encoder_state_1, encoder_output_2,
                                                                output_size = params['num_units'])

        return encoder_outputs, encoder_state


    def encode_Combine(self, encoder_output_1, encoder_state_1, encoder_output_2, output_size, dtype = tf.float32):
        

        with tf.name_scope("encoder_state_merge"):
            joined_state = tf.concat([encoder_state_1, encoder_output_2], axis = 1, name='concat')
            with tf.variable_scope("char_encoder"):
                W = tf.get_variable("join_W", [joined_state.get_shape()[1], output_size], dtype)
                b = tf.get_variable("join_b", [output_size], dtype)
            joined_state = tf.nn.xw_plus_b(joined_state, W, b, name="states_merged")
        encoder_state = joined_state
        encoder_output =encoder_output_1

        return encoder_output, encoder_state


    def cnn_encoder_output(self, encoder_emb_inp,mode , output_shape, num_filters, filter_sizes, embedding_size, sequence_length, dropout,dtype = tf.float32):

        embedded_chars_expanded = tf.expand_dims(encoder_emb_inp, -1)# LAst 1 As filter size
        dropout = dropout if mode == 'train' else 0.0
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []



        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1 , 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                
                pooled_outputs.append(pooled)

               

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis = 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        
        
        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob =(1.0 - dropout))

        with tf.name_scope("output_encoder_cnn"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, output_shape],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[output_shape]), name="b")
            output = tf.nn.xw_plus_b(h_drop, W, b, name="output")
        return output

    def _get_maximum_iterations(self, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if params['tgt_max_len_infer']:
            maximum_iterations = params['tgt_max_len_infer']
            utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * params["decoding_length_factor"]))
        return maximum_iterations

    def _build_decoder(self, encoder_outputs, encoder_state):
        """
        Build and run a RNN decoder with a final projection layer.
        Args:
            encoder_outputs: The outputs of encoder for every time step.
            encoder_state: The final state of the encoder.
        Returns:
            A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """
        tgt_sos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(params['sos'])), tf.int32)
        tgt_eos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(params['eos'])), tf.int32)

        num_layers = params['num_layers']

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_maximum_iterations(self.source_sequence_length)

        ##Decoder
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(encoder_outputs, encoder_state, self.source_sequence_length)

            # Train
            if self.mode != 'infer':
                # decoder_emp_inp: [max_time, batch_size, num_units]
                dec_input = self.src_input_ids
                if self.time_major:
                    dec_input = tf.transpose(dec_input)
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_src_decoder, dec_input)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.source_sequence_length,time_major=self.time_major)
                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper, decoder_initial_state)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=self.time_major,
                                                                                    swap_memory=True,scope=decoder_scope)

                sample_id = outputs.sample_id

                # Note: there's a subtle difference here between train and inference.
                # We could have set output_layer when create my_decoder
                #   and shared more code between train and inference.
                # We chose to apply the output_layer to all timesteps for speed:
                #   10% improvements for small models & 20% for larger ones.
                # If memory is a concern, we should apply output_layer per timestep.
                logits = self.output_layer(outputs.rnn_output)

            ## Inference
            else:

                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                # Helper
         



                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_src_decoder, start_tokens, end_token)

                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             helper,
                                                             decoder_initial_state,
                                                             output_layer=self.output_layer  # applied per timestep
                                                             )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                    maximum_iterations=maximum_iterations,
                                                                                    output_time_major=self.time_major,
                                                                                    swap_memory=True,
                                                                                    scope=decoder_scope)

                
                logits = self.output_layer(outputs.rnn_output)
                sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def _build_encoder_rnn_cell(self, params, num_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return create_rnn_cell(
                unit_type = params['unit_type'],
                num_units=params['num_units'],
                num_layers=num_layers,
                forget_bias=params['forget_bias'],
                dropout=params['dropout'],
                mode=self.mode)


    def _build_decoder_cell(self, encoder_outputs, encoder_state,source_sequence_length):
        """Build an RNN cell that can be used by decoder."""

        cell = create_rnn_cell( unit_type=params['unit_type'],
                                num_units=params['num_units'],
                                num_layers=params['num_layers'],
                                forget_bias=params['forget_bias'],
                                dropout=params['dropout'],
                                mode=self.mode)

        decoder_initial_state = encoder_state

        return cell, decoder_initial_state


    def _compute_loss(self, logits):
        '''
        Args:
            logits: [batch_size, sequence_length, num_decoder_symbols]
        '''
        """Compute optimization loss."""

        target_output = self.tgt_output_ids
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)    
        target_weights = tf.sequence_mask(self.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)
        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        # loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)

        loss = tf.contrib.seq2seq.sequence_loss(logits, target_output, weights = target_weights, average_across_batch = True)
        return loss


    def set_data_placeholders(self, train_iterator, val_iterator):
        #char_src_ids, src_input_ids, tgt_output_ids, char_seq_len, src_seq_len, tgt_seq_len = batched_iterator.get_next()
        
        def f_train():
            return train_iterator.get_next()
        def f_val():
            return val_iterator.get_next()

        
        char_src_ids, src_input_ids, tgt_output_ids, char_seq_len, src_seq_len, tgt_seq_len = tf.cond(tf.equal(self.iter_type,
                                                                                                               tf.constant(1, dtype=tf.int32)), 
                                                                                                               f_train, f_val)

        self.char_src_ids  = char_src_ids
        self.src_input_ids = src_input_ids
        self.tgt_output_ids = tgt_output_ids
        self.char_sequence_length = char_seq_len
        self.source_sequence_length = src_seq_len
        self.target_sequence_length = tgt_seq_len


    def train(self, sess):
        assert self.mode == 'train'
        return sess.run([self.debug,
                         self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count],
                         feed_dict = {self.iter_type : 1})

        # results =  sess.run([self.debug])
        # print (np.array(results[0]).shape)
        # return results

    def dev(self, sess):
        # return sess.run([self.softmax_scores, 
        #                  self.train_loss,
        #                  self.tgt_output_ids,
        #                  self.predict_count,
        #                  self.train_summary,
        #                  self.global_step,
        #                  self.word_count],
        #                  feed_dict = {self.iter_type : 2})
        #k = sess.run([self.logits, self.tgt_output_ids], feed_dict = {self.iter_type : 2})
        # print (k[0].shape, k[1].shape)
        return sess.run([self.softmax_scores, 
                         self.train_loss,
                         self.tgt_output_ids,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count],
                         feed_dict = {self.iter_type : 2})

    def infer(self, sess):
        return sess.run([self.logits])


    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


def create_rnn_cell(unit_type, num_units, num_layers,forget_bias, dropout, mode):
        """
        Create multi-layer RNN cell.
        Args:
            unit_type: string representing the unit type, i.e. "lstm".
            num_units: the depth of each unit.
            num_layers: number of cells.
            forget_bias: the initial forget bias of the RNNCell(s).
            dropout: floating point value between 0.0 and 1.0:
            the probability of dropout.  this is ignored if `mode != TRAIN`.
            mode: either tf.contrib.learn.TRAIN/EVAL/INFER
            num_gpus: The number of gpus to use when performing round-robin
            placement of layers.
            base_gpu: The gpu device id to use for the first RNN cell in the
            returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
            as its device id.

        Returns:
            An `RNNCell` instance.
        """
        cell_list = []
        for i in range(num_layers):
            utils.print_out("  cell %d" % i, new_line=False)
            single_cell = single_cell_fn(unit_type=unit_type,
                                         num_units=num_units,
                                         forget_bias=forget_bias,
                                         dropout=dropout,
                                         mode=mode,)
            utils.print_out("")
            cell_list.append(single_cell)


        if len(cell_list) == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)


def single_cell_fn(unit_type, num_units, forget_bias, dropout, mode, device_str=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == 'train' else 0.0

    # Cell Type
    if unit_type == "lstm":
        utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
        utils.print_out("  GRU", new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,new_line=False)
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units,forget_bias=forget_bias,layer_norm=True)
    elif unit_type == "nas":
        utils.print_out("  NASCell", new_line=False)
        single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),new_line=False)

    # Device Wrapper: Operator that ensures an RNNCell runs on a particular device.
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        utils.print_out("  %s, device=%s" %(type(single_cell).__name__, device_str), new_line=False)

    return single_cell
