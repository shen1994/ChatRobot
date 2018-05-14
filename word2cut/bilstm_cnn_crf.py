# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:18:27 2018

@author: shen1994
"""

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import ZeroPadding1D
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import merge

from keras_contrib.layers import CRF

from keras.models import Model

def bilstm_cnn_crf(maxlen, useful_word_len, class_label_count, embedding_size, embedding_weights=None, is_train=True):
    word_input = Input(shape=(maxlen,), dtype="int32", name="word_input")
        
    if is_train:
        word_emb = Embedding(useful_word_len, output_dim=embedding_size,
                                 input_length=maxlen, weights=[embedding_weights], \
                                 name="word_emb")(word_input)
    else:
        word_emb = Embedding(useful_word_len, output_dim=embedding_size, 
                                 input_length=maxlen, \
                                 name="word_emb")(word_input)
        
    # bilstm
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(word_emb)
    bilstm_drop = Dropout(0.1)(bilstm)
    bilstm_dense = TimeDistributed(Dense(embedding_size))(bilstm_drop)
        
    #cnn
    half_window_size = 2
    filter_kernel_number = 64
    padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)
    conv = Conv1D(nb_filter=filter_kernel_number, filter_length=2 * half_window_size + 1, padding="valid")(padding_layer)
    conv_drop = Dropout(0.1)(conv)
    conv_dense = TimeDistributed(Dense(filter_kernel_number))(conv_drop)
        
    #merge
    rnn_cnn_merge = merge([bilstm_dense, conv_dense], mode="concat", concat_axis=2)
    dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)
        
    # crf
    crf = CRF(class_label_count, sparse_target=False)
    crf_output = crf(dense)
        
    # mdoel
    model = Model(input=[word_input], output=crf_output)
    model.compile(loss=crf.loss_function, optimizer="adam", metrics=[crf.accuracy])
        
    return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        