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
            err = np.square(np.zeros(data_process.dec_embedding_length) - vec)
            mse = np.sum(err) / len(err)
            dec_dis_list.append(mse)
            for dec_word in dec_useful_words:

                std_number = np.std(dec_vec_model.wv[dec_word])
                if (std_number - data_process.epsilon) < 0:
                    norm_dec_vec = np.zeros(data_process.dec_embedding_length)
                else:
                    norm_dec_vec = (dec_vec_model.wv[dec_word] - np.mean(dec_vec_model.wv[dec_word])) / std_number
                err = np.square(norm_dec_vec - vec)
                mse = np.sum(err) / len(err)
                dec_dis_list.append(mse)
            index_list = np.argsort(dec_dis_list)
            index = index_list[1]
            if index == 0:
                word = data_process.__VOCAB__[0]
            else:
                word = dec_useful_words[index - 1]
            prediction_words.append(word)
        prediction_words_list.append(prediction_words)
        
    return prediction_words_list
    
def get_real_embedding(text_list):
    data_process = DataProcess(use_word2cut=True)
    dec_vocab = data_process.read_vocabulary(data_process.dec_vocab_file)
    
    dec_padding_ids_list = []

    for text in text_list:
    
        words = data_process.text_cut_object.cut([text.strip()])
        words_list = words[0].strip().split()
    
        dec_ids = [dec_vocab.get(word, data_process.__UNK__) for word in words_list]
    
        if len(dec_ids) + 2 > data_process.dec_output_length:
            dec_ids = dec_ids[:data_process.dec_output_length - 2]
            
        dec_length = len(dec_ids)
    
        dec_padding_ids = []
        dec_padding_ids.extend([data_process.__GO__])
        dec_padding_ids.extend([int(dec_ids[l]) for l in range(dec_length)])
        dec_padding_ids.extend([data_process.__EOS__])
        dec_padding_ids.extend([0] * (data_process.dec_output_length - dec_length - 2))
        
        dec_padding_ids_list.append(np.array(dec_padding_ids))
    
    padding_ids = np.array(dec_padding_ids_list)
    
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    dec_useful_words = list(dec_vec_model.wv.vocab.keys())
    dec_reverse_vec = data_process.read_reverse_vocabulary(data_process.dec_vocab_file)
    
    all_dec_embedding = []
    for one_padding_ids in padding_ids:
    
        dec_embedding = []
        for data in one_padding_ids:
            word = dec_reverse_vec[data]
            if word in dec_useful_words:
                word_embedding = dec_vec_model.wv[word]
            elif word == data_process.__VOCAB__[0]:
                word_embedding = np.zeros(data_process.dec_embedding_length)
            else:
                word_embedding = np.array([1.0] * data_process.dec_embedding_length)
            dec_embedding.append(word_embedding)
        all_dec_embedding.append(dec_embedding)
    
    return np.array([all_dec_embedding])

def run():

    model = build_model(training=False)
    
    model.load_weights("model/seq2seq_model_weights.h5")
    
    text = u"我真的好喜欢你，你认为呢？"

    enc_padding_ids = data_to_padding_ids([text])
    
    prediction_words = predict_text(model, enc_padding_ids)
    
    print(prediction_words)
    
if __name__ == "__main__":
    run()
    