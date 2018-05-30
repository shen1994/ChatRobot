# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:43:13 2018

@author: shen1994
"""

import gensim
import numpy as np

from data_process import DataProcess
from test import load_model
from test import data_to_padding_ids

def generate_real_embedding(text_list):
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
    
    return np.array(all_dec_embedding)
    
def run():
    questions = [u"我真的好喜欢你，你认为呢？", u"品尝大董意境菜时兴奋不已，并起身激情拥抱"]
    answers = [u"我也非常喜欢你。", "这个瞬间捕捉得很妙啊。"]
    model = load_model("model/seq2seq_model_weights.h5")
    enc_padding_ids = data_to_padding_ids(questions)
    prediction_embedding = model.predict_on_batch(enc_padding_ids)
    real_embedding = generate_real_embedding(answers)
    
    average_mse_list = []
    for pre, real in zip(prediction_embedding, real_embedding):
        error = pre - real
        square_error = np.square(error)
        square_sum_error = np.sum(square_error, axis=-1)
        average_mse = np.sum(square_sum_error) / len(square_sum_error)
        average_mse_list.append(average_mse)
    
    print("score: " + str(average_mse_list) + " ---> 0")         
    
if __name__ == "__main__":
    run()