from __future__ import print_function

params =\
{
    

    # # Misc Params
    "mode": 'train', # Choices: train, infer
    "create_new_model" : True, # False: use checkpoint
    "log_device_placement" : True,
    "sos":"<s>",#Start of sentence at decoder, cant be left blank
    "eos":"</s>",#End of sentence at decoder, cant be left blank
    "unk": "<unk>",# Unknown tag
    "char_pad": '<PAD>', # Check in char file
    "check_special_token":True,

    # Network Params
    "src_max_len":22,
    "tgt_max_len":22,
    "char_max_len" : 180, #cant be empty
    "tgt_max_len_infer" : None,
    "decoding_length_factor": 1.0,

    "num_epochs" : 4,
    "steps_per_devRun" : 1,# How many training steps to do per stats logging.
    "steps_per_stats" : 1,
    
    "colocate_gradients_with_ops":True, # Whether try colocating gradients with corresponding op

    "max_train" : 0,#Limit on the size of training data (0: no limit).
    "learning_rate": 1e-3,
    "num_buckets":1, #Put data into similar-length buckets.

    "time_major" : False, # Whether to use time-major mode for dynamic RNN.
    "num_layers" :1,
    "unit_type" :"gru",#lstm | gru | layer_norm_lstm | nas
    "forget_bias":1,#Forget bias for BasicLSTMCell
    "dropout": 0.2, #Dropout rate (not keep_prob)
    "max_gradient_norm" : 5.0, #Clip gradients to this norm.
    "num_units": 256, # Network size. enc, dec
    "share_vocab":False, # Share vocab table between source and target

    "num_filters_cnn": 5,
    "filter_sizes": [2,3,4],
    "l2_reg_lambda": 0.0,


    "optimizer" : 'sgd', #sgd | adam
    "batch_size": 128,


    # initializer weights params
    "init_weight" : 0.1, # for uniform init_op, initialize weights between [-this, this]."
    "init_op" : "uniform", # uniform | glorot_normal | glorot_uniform

    # File Location params
    "out_dir" : "runs", #Store log/model files.
    "desc_col" : "tokenized_desc",
    "label_col" : "yodlee_name",

    "enc_char_map_path" : "model_files/char_vocab.vocab", # Specify to overwrite
    "src_data_file": 'D:/TDE/CharEncoder-WordDecoderS2S/data/srcData.txt',# Data file with only sentences of source language, cant be empty in training
    "tgt_data_file": 'D:/TDE/CharEncoder-WordDecoderS2S/data/tgtData.txt',# Data file with only sentences of tgt language, cant be empty in training
    "src_vocab_file" : 'model_files/srcVocab.vocab',#Vocab file for src, created from data file if file doesnt exist, path should be correct
    "tgt_vocab_file" : 'model_files/tgtVocab.vocab',#Vocab file for tgt, created from data file

    "dev_src_file":'D:/TDE/CharEncoder-WordDecoderS2S/data/srcDevData.txt', # Validation data file, cant be empty
    "dev_tgt_file":'D:/TDE/CharEncoder-WordDecoderS2S/data/tgtDevData.txt', # Validation data file, cant be empty

    "test_src_file":'D:/TDE/CharEncoder-WordDecoderS2S/data/srcDevData.txt', # TEst data file, cant be empty
    "test_tgt_file":'D:/TDE/CharEncoder-WordDecoderS2S/data/tgtDevData.txt', # TEst data file, cant be empty

    "log_file" : "logs/log",

    # Inference time params
    "inference_input_file" : '', # Set to the text to decode
    "inference_output_file" : '', # Output file to store decoding results.
    'inference_ref_file' : '',# Reference file to compute evaluation scores (if provided)
    "ckpt" : '', # Checkpoint file to load a model for inference.
    "infer_batch_size" : 32, # Batch size for inference mode

}