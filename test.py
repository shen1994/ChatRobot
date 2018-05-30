# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:26:42 2018

@author: shen1994
"""

import gensim
import numpy as np

from encoder2decoder import build_model
from data_process import DataProcess

def data_to_padding_ids(text_list):
    
    data_process = DataProcess(use_word2cut=True)
    enc_vocab = data_process.read_vocabulary(data_process.enc_vocab_file)
    
    enc_padding_ids_list = []

    for text in text_list:
    
        words = data_process.text_cut_object.cut([text.strip()])
        words_list = words[0].strip().split()
    
        enc_ids = [enc_vocab.get(word, data_process.__UNK__) for word in words_list]
    
        if len(enc_ids) > data_process.enc_input_length:
            enc_ids = enc_ids[:data_process.enc_input_length]
            
        enc_length = len(enc_ids)
    
        enc_padding_ids = []
        enc_padding_ids.extend([0] * (data_process.enc_input_length - enc_length))
        enc_padding_ids.extend([int(enc_ids[enc_length - l - 1]) for l in range(enc_length)])
        
        enc_padding_ids_list.append(np.array(enc_padding_ids))
    
    return np.array(enc_padding_ids_list)
    
def calculate_mse(src_vec, des_vec):
    data_process = DataProcess(use_word2cut=False)
    
    std_number = np.std(des_vec)
    if (std_number - data_process.epsilon) < 0:
        norm_des_vec = np.zeros(data_process.dec_embedding_length)
    else:
        norm_des_vec = (des_vec - np.mean(des_vec)) / std_number
                
    err = np.square(src_vec - norm_des_vec)
    mse = np.sum(err)
    
    return mse
    
def predict_text(model, enc_embedding):
    
    data_process = DataProcess(use_word2cut=False)
    
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    dec_useful_words = tuple(dec_vec_model.wv.vocab.keys())
    
    prediction = model.predict_on_batch(enc_embedding)
    
    prediction_words_list = []
    for elem in prediction:
        prediction_words = []
        for vec in elem:
            dec_dis_list = []
            mse = calculate_mse(vec, np.zeros(data_process.dec_embedding_length))
            dec_dis_list.append(mse)
            for dec_word in dec_useful_words:
                mse = calculate_mse(vec, dec_vec_model.wv[dec_word]) 
                dec_dis_list.append(mse)
            index = np.argmin(dec_dis_list)
            if index == 0:
                word = data_process.__VOCAB__[0]
            else:
                word = dec_useful_words[index - 1]
            prediction_words.append(word)
        prediction_words_list.append(prediction_words)
        
    return prediction_words_list
    
def load_model(model_path):
    
    model = build_model(training=False)
    
    model.load_weights(model_path)
    
    return model
    
def common_prediction(model, text):

    padding_ids = data_to_padding_ids(text)
    
    words = predict_text(model, padding_ids)
    
    return words

def run():
    
    text = [u"我真的好喜欢你，你认为呢？"]
    
    model = load_model("model/seq2seq_model_weights.h5")

    prediction_words = common_prediction(model, text)
    
    print(prediction_words)
    
if __name__ == "__main__":
    run()
    