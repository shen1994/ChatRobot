# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:23:57 2018

@author: shen1994
"""

import codecs
import numpy as np

from data_process import DataProcess

def generate_batch(batch_size=None, encoder_file_path=None, decoder_file_path=None, \
                   encoder_word2vec_model=None, decoder_word2vec_model=None, embedding_shape=None):
    
    if not encoder_word2vec_model or not decoder_word2vec_model or not embedding_shape:
        print(u"未加载中文词向量模型")
        return
        
    data_process = DataProcess(use_word2cut=False)
    enc_reverse_vec = data_process.read_reverse_vocabulary(data_process.enc_vocab_file)
    dec_reverse_vec = data_process.read_reverse_vocabulary(data_process.dec_vocab_file)
    enc_useful_words = list(encoder_word2vec_model.wv.vocab.keys())
    dec_useful_words = list(decoder_word2vec_model.wv.vocab.keys())
    
    batch_count = 0
    
    X = []
    Y = []
    
    while True:
        
        source_index_padding = codecs.open(encoder_file_path, "r", "utf-8")
        target_index_padding = codecs.open(decoder_file_path, "r", "utf-8")
        
        source_line = source_index_padding.readline()
        target_line = target_index_padding.readline()
        
        while source_line and target_line:
        
            source_str_list = source_line.strip().split()
            target_str_list = target_line.strip().split()
            
            source_list = []
            for data in source_str_list:
                word = enc_reverse_vec[int(data)]
                if word in enc_useful_words:
                    word_embedding = encoder_word2vec_model.wv[word]
                elif word == data_process.__VOCAB__[0]:
                    word_embedding = np.zeros(embedding_shape[0])
                else:
                    word_embedding = np.random.uniform(-1, 1, embedding_shape[0])
                source_list.append(word_embedding)
            
            target_list = []
            for data in target_str_list:
                word = dec_reverse_vec[int(data)]
                if word in dec_useful_words:
                    word_embedding = decoder_word2vec_model.wv[word]
                elif word == data_process.__VOCAB__[0]:
                    word_embedding = np.zeros(embedding_shape[1])
                else:
                    word_embedding = np.random.uniform(-1, 1, embedding_shape[1])
                target_list.append(word_embedding)
    
            X.append(source_list)
            Y.append(target_list)
            
            batch_count += 1
            
            if batch_count == batch_size:
                
                batch_count = 0
                
                X_ARRAY = np.array(X)
                Y_ARRAY = np.array(Y)
                
                yield(X_ARRAY, Y_ARRAY)
                
                X = []
                Y = []
            
            source_line = source_index_padding.readline()
            target_line = target_index_padding.readline()
            
        source_index_padding.close()
        target_index_padding.close()    
        