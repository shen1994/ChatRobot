# -*- coding: utf-8 -*-
"""
Created on Sun May  6 21:25:48 2018

@author: shen1994
"""

import os
import pickle

import numpy as np

from .fake_keras import pad_sequences
from .bilstm_cnn_crf import bilstm_cnn_crf

class WordCut:
    
    def __init__(self):
        
        model, lexicon, sequence_max_length = self.load_params()
        
        self.model = model
        self.lexicon = lexicon
        self.sequence_max_length = sequence_max_length

    def predict_one_text(self, text, label):
        
        start_index = len(label) - len(text)
        
        label = label[start_index:]
    
        segment_text = ""
        for p, t in zip(label, text):
            if p in [0, 3, 4, 5]:
                segment_text += (t + " ")
            else:
                segment_text += t
                
        return segment_text
        
    def predict_many_text(self, text_list, model, lexicon, maxlen):
        
        new_text_list = []
    
        for text in text_list:
            temp = []
            for c in text:
                if c in lexicon:
                    temp.append(lexicon.get(c))
                else:
                    temp.append(-1)
            new_text_list.append(temp)
            
        test_array = pad_sequences(new_text_list, maxlen=maxlen)
        
        test_pred = model.predict(test_array, verbose=0)
        
        label_list = np.argmax(test_pred,axis=2)
        
        new_text_list = []
        for text, label in zip(text_list, label_list):
            new_text = self.predict_one_text(text, label)
            new_text_list.append(new_text)
            
        return new_text_list
        
    def load_params(self):

        model_path = os.getcwd().replace("\\", "/")
        model_path = model_path + os.sep + 'word2cut' + os.sep
        
        sequence_max_length, embedding_size, \
        useful_word_length, label_2_index_length = pickle.load(open(model_path + 'model/model_params.pkl','rb'))
        
        model = bilstm_cnn_crf(sequence_max_length, useful_word_length,\
                               label_2_index_length, embedding_size, is_train=False)
        
        model.load_weights(model_path + 'model/train_model.hdf5')
        
        lexicon, index_2_label = pickle.load(open(model_path + 'model/lexicon.pkl','rb'))
        
        return model, lexicon, sequence_max_length
        
    def cut(self, text_list):
                
        new_text_list = self.predict_many_text(text_list, self.model, self.lexicon, self.sequence_max_length)
                
        return new_text_list
    