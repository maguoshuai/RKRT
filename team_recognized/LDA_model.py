# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 下午3:55
# @Author  : nevermore.huachi
# @File    : LDA_model.py
# @Software: PyCharm

from gensim.models.ldamodel import LdaModel
from nltk.corpus import wordnet as wn
from gensim.corpora.dictionary import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pickle
import numpy as np

# 第一次使用需要首先下载en包:
# python -m spacy download en
import spacy

# 第一次使用需要下载停顿词
# nltk.download('stopwords')
# 英文停顿词停顿词处理
en_stop = set(nltk.corpus.stopwords.words('english'))

# spacy.load('en_core_web_sm')
# from spacy.lang.en import English
# parser = English()


# 对文章内容进行清洗并将单词统一降为小写
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


# 定义预处理函数
def prepare_text_for_lda(text):
    # 分词处理
    tokens = tokenize(text)
    # 取出长度大于4的单词
    tokens = [token for token in tokens if len(token) > 4]
    # 取出非停顿词
    tokens = [token for token in tokens if token not in en_stop]
    # 对词语进行还原
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# 获取每个作者的LDA向量
def lda_vector(field, db_col):
    print('++++++++++++++++ running  ' + field + ' lda model ++++++++++++++++++++')
    author_paper_l = []
    author_list = []
    for author in db_col.find():
        author_list.append(author["id"])
        papers_des = []
        for paper in author['pub_papers']:
            if paper['description']:
                papers_des += prepare_text_for_lda(paper['description'].strip())
        author_paper_l.append(papers_des)
    common_dictionary = Dictionary(author_paper_l)
    common_corpus = [common_dictionary.doc2bow(text) for text in author_paper_l]
    ldamodel = LdaModel(common_corpus, id2word=common_dictionary, alpha='auto', eval_every=5)
    # vec_mat = sparse.dok_matrix((1, ldamodel.num_topics))
    author_ldavec = {}
    for author, vec in zip(author_list, ldamodel.get_document_topics(common_corpus)):
        ldavec = np.zeros(ldamodel.num_topics)
        for ve in vec:
            ldavec[ve[0]] = ve[1]
        author_ldavec[author] = ldavec

    print("正在保存文本向量.......")
    ldavec_path = '../lda_model/' + field + 'lda.pickle'
    with open(ldavec_path, 'wb') as fp:
        pickle.dump(author_ldavec, fp)
    print("+++++++++++++++++++++++++++文本向量保存完成++++++++++++++++++++++++++++++++")