# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:26:42 2018

@author: shen1994
"""

import gensim
import numpy as np

from keras.utils import plot_model
from seq2seq import AttentionSeq2Seq
from data_process import DataProcess

def data_to_padding_ids(text):
    data_process = DataProcess(use_word2cut=True)
    
    words = data_process.text_cut_object.cut([text.strip()])
    words_list = words[0].strip().split()
    
    enc_vocab = data_process.read_vocabulary(data_process.enc_vocab_file)
    
    enc_ids = [enc_vocab.get(word, data_process.__UNK__) for word in words_list]
    
    if len(enc_ids) > data_process.enc_input_length:
                enc_ids = enc_ids[:data_process.enc_input_length]
            
    enc_length = len(enc_ids)
    
    enc_padding_ids = []
    enc_padding_ids.extend([0] * (data_process.enc_input_length - enc_length))
    enc_padding_ids.extend([int(enc_ids[enc_length - l - 1]) for l in range(enc_length)])
    
    return enc_padding_ids
    
def data_to_embedding(enc_padding_ids):
    
    data_process = DataProcess(use_word2cut=False)
    
    enc_vec_model = gensim.models.Word2Vec.load(r'model/encoder_vector.m')
    enc_useful_words = list(enc_vec_model.wv.vocab.keys())
    enc_reverse_vec = data_process.read_reverse_vocabulary(data_process.enc_vocab_file)
    
    enc_embedding = []
    for data in enc_padding_ids:
        word = enc_reverse_vec[data]
        if word in enc_useful_words:
            word_embedding = enc_vec_model.wv[word]
        elif word == data_process.__VOCAB__[0]:
            word_embedding = np.zeros(data_process.enc_embedding_length)
        else:
            word_embedding = np.random.uniform(-1, 1, data_process.enc_embedding_length)
        enc_embedding.append(word_embedding)
    
    return np.array([enc_embedding])
    
def predict_one_text(model, enc_embedding):
    
    data_process = DataProcess(use_word2cut=False)
    
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    dec_useful_words = list(dec_vec_model.wv.vocab.keys())
    
    prediction = model.predict(enc_embedding, verbose=0)   
    
    prediction_words = []
    for vec in prediction[0]:
        dec_dis_list = []
        dec_dis = np.sqrt(np.sum(np.square(np.zeros(data_process.dec_embedding_length) - vec)))
        dec_dis_list.append(dec_dis)
        for dec_word in dec_useful_words:
            dec_dis = np.sqrt(np.sum(np.square(dec_vec_model.wv[dec_word] - vec)))
            dec_dis_list.append(dec_dis)
        index = np.argmin(dec_dis_list)
        if index == 0:
            word = data_process.__VOCAB__[0]
        else:
            word = dec_useful_words[index - 1]
        prediction_words.append(word)
        
    return prediction_words
        
def print_score(model, enc_embedding):
    data_process = DataProcess(use_word2cut=False)
    
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    dec_useful_words = list(dec_vec_model.wv.vocab.keys())
    prediction = model.predict(enc_embedding, verbose=0)
    
    score_words = []
    
    for vec in prediction[0]:
        dec_sum = 0
        dec_dis_list = []
        dec_dis = np.sqrt(np.sum(np.square(np.zeros(data_process.dec_embedding_length) - vec)))
        dec_dis_list.append(dec_dis)
        dec_sum += dec_dis
        for dec_word in dec_useful_words:
            dec_dis = np.sqrt(np.sum(np.square(dec_vec_model.wv[dec_word] - vec)))
            dec_dis_list.append(dec_dis)
            dec_sum += dec_dis
        score_words.append(dec_dis_list / dec_sum)
    
    print(score_words)

def run():
    
    data_process = DataProcess(use_word2cut=False)

    input_length = data_process.enc_input_length
    output_length = data_process.dec_output_length
    enc_embedding_length = data_process.enc_embedding_length
    dec_embedding_length = data_process.dec_embedding_length
    
    model = AttentionSeq2Seq(output_dim=dec_embedding_length, hidden_dim=data_process.hidden_dim, output_length=output_length, \
                             input_shape=(input_length, enc_embedding_length), \
                             batch_size=1, \
                             depth=data_process.layer_shape)

    model.compile(loss='mse', optimizer='rmsprop')
    
    model.load_weights("model/seq2seq_model_weights.h5")
    
    plot_model(model, to_file='model/seq2seq_model_structure.png', show_shapes=True, show_layer_names=True)
    
    text = u"碧水照嫩柳，桃花映春色"#u"这围巾要火！"#u"你愿意嫁给我吗？"

    enc_padding_ids = data_to_padding_ids(text)
    enc_embedding = data_to_embedding(enc_padding_ids)
    
    prediction_words = predict_one_text(model, enc_embedding)
    
    print(prediction_words)
    
    print_score(model, enc_embedding)
    
if __name__ == "__main__":
    run()