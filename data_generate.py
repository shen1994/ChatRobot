# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:23:57 2018

@author: shen1994
"""

import codecs
import gensim
import numpy as np

from data_process import DataProcess

def generate_batch(batch_size=None):
        
    data_process = DataProcess(use_word2cut=False)
    
    
    dec_reverse_vec = data_process.read_reverse_vocabulary(data_process.dec_vocab_file)
    decoder_word2vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    dec_useful_words = list(decoder_word2vec_model.wv.vocab.keys())
    
    batch_count = 0
    
    X = []
    Y = []
    
    while True:
        
        source_index_padding = codecs.open(data_process.enc_ids_padding_file, "r", "utf-8")
        target_index_padding = codecs.open(data_process.dec_ids_padding_file, "r", "utf-8")
        
        source_line = source_index_padding.readline()
        target_line = target_index_padding.readline()
        
        while source_line and target_line:
        
            source_str_list = source_line.strip().split()
            target_str_list = target_line.strip().split()
            
            source_list = []
            for elem in source_str_list:
                source_list.append(int(elem))
            
            target_list = []
            for data in target_str_list:
                word = dec_reverse_vec[int(data)]
                if word in dec_useful_words:
                    word_embedding = decoder_word2vec_model.wv[word]
                elif word == data_process.__VOCAB__[0]:
                    word_embedding = np.zeros(data_process.dec_embedding_length)
                else:
                    word_embedding = np.array([1.0] * data_process.dec_embedding_length)
                
                # normalization
                std_number = np.std(word_embedding)
                if (std_number - data_process.epsilon) < 0:
                    word_embedding_scale = np.zeros(data_process.dec_embedding_length)
                else:
                    word_embedding_scale = (word_embedding - np.mean(word_embedding)) / std_number

                target_list.append(word_embedding_scale)
    
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
        