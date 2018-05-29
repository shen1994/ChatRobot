# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:04:46 2018

@author: shen1994
"""

import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_with_lables(X, Y, filename="model/tsne.png"):
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(Y):
        x, y = X[i, :]
        plt.scatter(x, y)
        plt.annotate(label, 
                     xy=(x, y), 
                     xytext=(5,2), 
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

if __name__ == "__main__":
    dec_vec_model = gensim.models.Word2Vec.load(r'model/decoder_vector.m')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    dec_useful_words = list(dec_vec_model.wv.vocab.keys())
    
    plot_only=200
    
    X = []
    Y_LABELS = []
    counter = 0
    for elem in dec_useful_words:
        X.append(dec_vec_model.wv[elem])
        Y_LABELS.append(elem.encode("utf8"))
        if counter > plot_only:
            break
        counter += 1
    X_TRANS = tsne.fit_transform(X)
    
    plot_with_lables(X_TRANS, Y_LABELS)