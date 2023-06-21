from gensim.models import Word2Vec,FastText
# import pandas as pd
# import re
from matplotlib import pyplot
from sklearn.decomposition import PCA
# from compue import read_file
# from matplotlib import pyplot as plt
# # import plotly.graph_objects as go
#
# import numpy as np

# from compue import vname
vname = 'v3'

def chunk2vec(file_name,out_name):
    with open(file_name) as f:
        lines = f.readlines()
    sentences = []
    for line in lines:
        words = line.strip().split(" ")
        sentences.append(words)

    # train model
    model = Word2Vec(sentences,
                     size=10,
                     window=3,
                     iter=50,
                     sg=0,
                     min_count=5
                     )
    # fit a 2d PCA model to the vectors
    model.save(f"model/{vname}/word2vec/{out_name}.model")
    # model.wv.save_word2vec_format(f"model/{vname}/csdn_vector.txt", binary=False)
    # show(model)

def chunk2vec_fasttext(file_name,size,window,min_count,iter,out_name):
    with open(file_name) as f:
        lines = f.readlines()
    sentences = []
    for line in lines:
        words = line.strip().split(" ")
        sentences.append(words)

    # train model
    model = FastText(sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     iter=iter
                     )
    # fit a 2d PCA model to the vectors
    model.save(f"model/{vname}/fasttext/{out_name}.model")
    # model.wv.save_word2vec_format(f"model/{vname}/fasttext/csdn_vector_fasttext-{out_name}.txt", binary=False)
    # show(model)


def show(model):
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        # $ 显示会出问题
        if '$' in word:
            continue
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


if __name__ == '__main__':

    # 训练模型
    # chunk2vec(f'data/{vname}/chunk_csdn.txt')
    chunk2vec_fasttext(f'data/{vname}/chunk_csdn_level.txt', 10, 3, 50, 5, 'csdn_fastext-s10-w3-e50-m5_level')
    # m = Word2Vec.load(f"model/{vname}/csdn.model")
    # show(m)
    # ws = '123 456 78 lai ming'.split(' ')

    # ws = read_file("chunk.txt", item_ind=1, n=1000)
    # for k in ws:
    #     s = m.wv.similar_by_word(k)
    #     print(k, s)
