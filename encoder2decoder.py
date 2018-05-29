# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:56:15 2018

@author: shen1994
"""

import gensim
import numpy as np

from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.models import Sequential

from seq2seq import AttentionSeq2Seq
from data_process import DataProcess

def get_encoder_embedding():
    
    embedding_list = []
    data_process = DataProcess(use_word2cut=False)
    vocab_dict = data_process.read_reverse_vocabulary(data_process.enc_vocab_file)
    vec_model = gensim.models.Word2Vec.load(r'model/encoder_vector.m')

    for key, value in vocab_dict.items():
        if key == data_process.__PAD__:
            embedding_list.append(np.array([0.0] * data_process.enc_embedding_length))
        elif key == data_process.__UNK__:
            embedding_list.append(np.array([1.0] * data_process.enc_embedding_length))
        else:
            embedding_list.append(vec_model.wv[value])
    
    return np.array(embedding_list)

def build_model(training=True):
    
    data_process = DataProcess(use_word2cut=False)
    
    embedding_matrix = get_encoder_embedding()
    vocab_size, embedding_size = embedding_matrix.shape
    embedding_layer = Embedding(
        vocab_size,
        embedding_size,
        weights=[embedding_matrix],
        input_length=data_process.enc_input_length,
        trainable=training,
        name='encoder_embedding')

    enc_normalization = BatchNormalization(epsilon=data_process.epsilon)

    seq2seq = AttentionSeq2Seq(
        bidirectional=False,
        output_dim=data_process.dec_embedding_length, 
        hidden_dim=data_process.hidden_dim, 
        output_length=data_process.dec_output_length, 
        input_shape=(data_process.enc_input_length, data_process.enc_embedding_length), 
        depth=data_process.layer_shape)
    
    model = Sequential()
    model.add(embedding_layer)
    model.add(enc_normalization)
    model.add(seq2seq)
    
    # from keras.optimizers import SGD
    # sgd = SGD(lr=0.001, decay=0, clipvalue=0.0)
    # from keras.optimizers import Adam
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-04)
    # adam rmsprop sgd
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    return model
    