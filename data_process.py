# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:02:57 2018

@author: shen1994
"""

import os
import codecs

from word2cut import WordCut

class DataProcess:
    __PAD__ = 0
    __UNK__ = 1
    __GO__ = 2
    __EOS__ = 3
    __VOCAB__ = ['__PAD__', '__UNK__', '__GO__', '__EOS__']

    def __init__(self, use_word2cut=True):
        
        self.corpus_path = "corpus"
        self.model_path = "model"
        self.data_path = "data"
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        self.enc_vocab_size = 20000
        self.dec_vocab_size = 20000
        self.enc_input_length = 50
        self.dec_output_length = 50
        self.enc_embedding_length = 128
        self.dec_embedding_length = 128
        self.hidden_dim = 100
        self.layer_shape = (2, 1)
        self.epsilon = 1e-6
        
        self.enc_file = self.corpus_path + os.sep + "question.txt"
        self.dec_file = self.corpus_path + os.sep + "answer.txt"
        self.enc_vocab_file = self.model_path + os.sep + \
                                "enc_vocab" + str(self.enc_vocab_size) + ".data"
        self.dec_vocab_file = self.model_path + os.sep + \
                                "dec_vocab" + str(self.dec_vocab_size) + ".data"
        self.enc_ids_file = self.data_path + os.sep + "enc_ids.data"
        self.dec_ids_file = self.data_path + os.sep + "dec_ids.data"
        self.enc_ids_padding_file = self.data_path + os.sep + "enc_padding_ids.data"
        self.dec_ids_padding_file = self.data_path + os.sep + "dec_padding_ids.data"
        
        if use_word2cut:
            self.text_cut_object = WordCut()
        
    def create_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size, mode=False):
        
        vocab = dict()
        vocab_list = list()

        with codecs.open(data_path, "r", "utf-8") as f:
            line = f.readline()
            while(line):
                
                line = line.strip()
                
                words = self.text_cut_object.cut([line])
                
                words_list = words[0].strip().split()               
                
                for word in words_list:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                
                line = f.readline()
        
        if mode:    
            vocab_list = self.__VOCAB__ + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab_list = [self.__VOCAB__[0]] + [self.__VOCAB__[1]] + sorted(vocab, key=vocab.get, reverse=True)
                
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
                
        with codecs.open(vocabulary_path, "w", "utf-8") as f:
            counter = 0
            for w in vocab_list:
                f.write(w + "\t" + str(counter) + "\n")
                counter += 1
                
    def read_vocabulary(self, vocabulary_path):
        vocab = dict()
        with codecs.open(vocabulary_path, "r", "utf-8") as f:
            line = f.readline()
            while(line):
                line_list = line.strip().split()
                vocab[line_list[0]] = int(line_list[1])
                line=f.readline()
                
        return vocab
        
    def read_reverse_vocabulary(self, vocabulary_path):
        vocab = dict()
        with codecs.open(vocabulary_path, "r", "utf-8") as f:
            line = f.readline()
            while(line):
                line_list = line.strip().split()
                vocab[line_list[0]] = int(line_list[1])
                line=f.readline()
                
        reverse_dict = dict(zip(vocab.values(), vocab.keys())) 
        
        return reverse_dict
                
    def sentence_to_ids(self, sentence, vocabulary):
        words = self.text_cut_object.cut([sentence.strip()])
        words_list = words[0].strip().split()
        return [vocabulary.get(word, self.__UNK__) for word in words_list]
                
    def data_to_ids(self, data_path, target_path, vocabulary):
        target_writer = codecs.open(target_path, "w", "utf-8")
        with codecs.open(data_path, "r", "utf-8") as f:
            line = f.readline()
            while(line):
                
                data_ids = self.sentence_to_ids(line, vocabulary)
                
                target_writer.write(" ".join([str(one_ids) for one_ids in data_ids]) + "\n")
                
                line = f.readline()
                
    def data_to_padding_ids(self, source_path, target_path, \
                              source_padding_path, target_padding_path, \
                              source_max_length=None, target_max_length=None):
        
        if not source_max_length or not target_max_length or source_max_length < 3 or target_max_length < 3:  
            print(u"未给数据最大长度或数据长度太短")
            return
        
        source_file = codecs.open(source_path, "r", "utf-8")
        target_file = codecs.open(target_path, "r", "utf-8")
        source_padding_writer = codecs.open(source_padding_path, "w", "utf-8")
        target_padding_writer = codecs.open(target_padding_path, "w", "utf-8")
        
        source = source_file.readline()
        target = target_file.readline()
        
        while source and target:
            
            source_line = source.strip().split()
            
            if len(source_line) > source_max_length:
                source_line = source_line[:source_max_length]
            
            source_length = len(source_line)
            
            source_ids = []
            source_ids.extend([0] * (source_max_length - source_length))
            source_ids.extend([int(source_line[source_length - l - 1]) for l in range(source_length)])
            
            target_line = target.strip().split()
            
            if len(target_line) + 2 > target_max_length:
                target_line = target_line[:target_max_length - 2]
            
            target_length = len(target_line)
    
            target_ids = []
            target_ids.append(self.__GO__)
            for x in target_line:
                target_ids.append(int(x))
            target_ids.append(self.__EOS__)
            target_ids.extend([0] * (target_max_length - target_length - 2))

            source_padding_writer.write(" ".join([str(src) for src in source_ids]) + "\n")
            target_padding_writer.write(" ".join([str(des) for des in target_ids]) + "\n")
                    
            source = source_file.readline()
            target = target_file.readline()
        
        source_file.close()
        target_file.close()
        source_padding_writer.close()
        target_padding_writer.close()
        
    def get_documents_size(self, source_path, target_path):
        
        source_file = codecs.open(source_path, "r", "utf-8")
        target_file = codecs.open(target_path, "r", "utf-8")
        
        source = source_file.readline()
        target = target_file.readline() 
        counter = 0
        while source and target:
            counter += 1
            source = source_file.readline()
            target = target_file.readline()
        source_file.close()
        target_file.close()
        
        return counter
                
    def run(self):
        self.create_vocabulary(self.enc_vocab_file, self.enc_file, self.enc_vocab_size)
        self.create_vocabulary(self.dec_vocab_file, self.dec_file, self.dec_vocab_size, mode=True)
        enc_vocab = self.read_vocabulary(self.enc_vocab_file)
        self.data_to_ids(self.enc_file, self.enc_ids_file, enc_vocab)
        dec_vocab = self.read_vocabulary(self.dec_vocab_file)
        self.data_to_ids(self.dec_file, self.dec_ids_file, dec_vocab)
        self.data_to_padding_ids(self.enc_ids_file, self.dec_ids_file, \
                                 self.enc_ids_padding_file, self.dec_ids_padding_file, \
                                 self.enc_input_length, self.dec_output_length)
        print(u"数据预处理完成" + "---OK")
        
if __name__ == "__main__":
    data_process = DataProcess()
    data_process.run()
        