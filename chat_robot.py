# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:47:57 2018

@author: shen1994
"""

from data_process import DataProcess
from test import load_model
from test import common_prediction

def clean_repeat_words(words):
    
    data_process = DataProcess(use_word2cut=False)
    
    words_length = len(words)
    
    if words_length < 2:
        return words
    
    repeat_words = ["，", "；", "。", "？", "！"]

    new_words = []
    last_word = words[0]
    if not (last_word == data_process.__VOCAB__[0]):
        new_words.append(last_word)
    for index in range(1, words_length):
        if (words[index] == last_word) and (words[index] in repeat_words):
            continue
        else:
            if not (last_word == data_process.__VOCAB__[0]):
                new_words.append(words[index])
            last_word = words[index]

    return new_words

def assembly_word(words):
    
    default_answer = u"小哥哥，对不起呢，我不知道。"
    
    data_process = DataProcess(use_word2cut=False)
    
    words_length = len(words)
    
    EOS_index = -1
    for index in range(words_length):
        if words[index] == data_process.__VOCAB__[3]:
            EOS_index = index
            break
        
    if EOS_index == 0 or EOS_index == -1:
        return default_answer
        
    GO_index = -1
    for index in range(words_length):
        if words[index] == data_process.__VOCAB__[2]:
            GO_index = index
            break
        
    new_words = []

    if (GO_index - EOS_index) >= -1:
        return default_answer
    
    if GO_index == -1:
        new_words.extend(words[0:EOS_index])
    else:
        new_words.extend(words[(GO_index + 1):EOS_index])
    
    new_words = clean_repeat_words(new_words)
    
    if not new_words:
        return default_answer
    
    text = "".join(word for word in new_words)
    
    return text

def run():

    questions = [u"我喜欢你？", u"品尝大董意境菜时兴奋不已，并起身激情拥抱"]

    model = load_model("model/seq2seq_model_weights.h5")

    prediction_words = common_prediction(model, questions)
    
    for index in range(len(questions)):
        
        print("------------------------------\n")
        
        print("问： " + questions[index])
    
        answer = assembly_word(prediction_words[index])

        print("答： " + answer)
        
    print("------------------------------\n")


if __name__ == "__main__":
    run()