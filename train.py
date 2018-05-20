# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:58:09 2018

@author: shen1994
"""

import gensim

from seq2seq import AttentionSeq2Seq
from data_process import DataProcess
from data_generate import generate_batch

def run():

    enc_vec_model = gensim.models.Word2Vec.load(r'model/encoder_vector.m')
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    
    batch_size = 9
    epochs = 30
    data_process = DataProcess(use_word2cut=False)
    documents_length = data_process.get_documents_size(data_process.enc_ids_file, data_process.dec_ids_file)
    input_length = data_process.enc_input_length
    output_length = data_process.dec_output_length
    enc_embedding_length = data_process.enc_embedding_length
    dec_embedding_length = data_process.dec_embedding_length
    
    if batch_size > documents_length:
        print("ERROR--->" + u"语料数据量过少，请再添加一些")
        return None
        
    if (data_process.hidden_dim < data_process.enc_input_length):
        print("ERROR--->" + u"隐层神经元数目过少，请再添加一些")
        return None
        
    model = AttentionSeq2Seq(output_dim=dec_embedding_length, hidden_dim=data_process.hidden_dim, output_length=output_length, \
                             input_shape=(input_length, enc_embedding_length), 
                             batch_size=batch_size, 
                             depth=data_process.layer_shape)
    # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit_generator(generator=generate_batch(batch_size=batch_size, \
                                                 encoder_word2vec_model=enc_vec_model, \
                                                 decoder_word2vec_model=dec_vec_model, \
                                                 encoder_file_path=data_process.enc_ids_padding_file, \
                                                 decoder_file_path=data_process.dec_ids_padding_file, \
                                                 embedding_shape = (enc_embedding_length, dec_embedding_length)),
                        steps_per_epoch=int(documents_length / batch_size), \
                        epochs=epochs, verbose=1, workers=1)
    
    model.save_weights("model/seq2seq_model_weights.h5", overwrite=True)
    
if __name__ == "__main__":
    run()
    
    