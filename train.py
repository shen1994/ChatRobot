# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:58:09 2018

@author: shen1994
"""

from data_process import DataProcess
from data_generate import generate_batch
from encoder2decoder import build_model

from test import data_to_padding_ids
from test import predict_text

def run():
    batch_size = 63
    epochs = 5000
    
    data_process = DataProcess(use_word2cut=False)

    model = build_model()
  
    documents_length = data_process.get_documents_size(data_process.enc_ids_file, data_process.dec_ids_file)
    
    if batch_size > documents_length:
        print("ERROR--->" + u"语料数据量过少，请再添加一些")
        return None

    model.fit_generator(generator=generate_batch(batch_size=batch_size),
                        steps_per_epoch=int(documents_length / batch_size), \
                        epochs=epochs, verbose=1, workers=1)

    model.save_weights("model/seq2seq_model_weights.h5", overwrite=True)
    
if __name__ == "__main__":
    run()
    
    