#/usr/bin/python
# -*- coding:utf-8 -*-
'''
Author: changjingdong
Date: 20190614
Desc: data processer for xgboost
'''
import sys
import codecs
from collections import defaultdict
from collections import Counter
import pandas as pd
import numpy as np
import functools
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import os
sys.path.insert(0, '../common')
from commen_function import *
os.environ['CUDA_VISIBLE_DEVICES']='2'

def sent2vec(s, stop_words, word2vec):
    word_list = remove_stopword(s.lower().split('|'), stop_words)
    M = list_conv(list_filter(word_list, lambda x: x in word2vec), lambda x: word2vec[x])
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

## basic features
def get_basic_feat(data, EMBEDDING_DIM, stop_words, word2vec):
    data['len_q1'] = data.question1.apply(lambda x: len(x.replace('|', '')))
    data['len_q2'] = data.question2.apply(lambda x: len(x.replace('|', '')))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(x.replace('|', '')))))
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(x.replace('|', '')))))
    data['len_word_q1'] = data.question1.apply(lambda x: len(x.split('|')))
    data['len_word_q2'] = data.question2.apply(lambda x: len(x.split('|')))
    data['common_words'] = data.apply(lambda x: len(set(x['question1'].lower().split('|')).intersection(set(x['question2'].lower().split('|')))), axis=1)

    question1_vectors = np.zeros((data.shape[0], EMBEDDING_DIM))
    for i, q in enumerate(data.question1.values):
        question1_vectors[i, :] = sent2vec(q, stop_words, word2vec)

    question2_vectors  = np.zeros((data.shape[0], EMBEDDING_DIM))
    for i, q in enumerate(data.question2.values):
        question2_vectors[i, :] = sent2vec(q, stop_words, word2vec)
    
    def feat_func(func):
        return [func(x,y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

    #data['jaccard_distance'] = feat_func(jaccard)
    #data['euclidean_distance'] = feat_func(euclidean)
    data['cosine_distance'] = feat_func(cosine)
    data['cityblock_distance'] = feat_func(cityblock)
    data['canberra_distance'] = feat_func(canberra)
    data['braycurtis_distance'] = feat_func(braycurtis)
    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

    data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    for dim in range(question1_vectors.shape[1]):
        data['q1_w2v_dim%d'%dim] = question1_vectors[:,dim]
    for dim in range(question2_vectors.shape[1]):
        data['q2_w2v_dim%d'%dim] = question2_vectors[:,dim]
    #data.drop(['question1', 'question2'], axis=1, inplace=True)
    
    return data

def remove_stopwords(s, stops):
    return list_filter(s, lambda x: x not in stops)

## tfidf features
def word_match_share(row, stops={}):
    q1words = remove_stopwords(row['question1'].split('|'), stops)
    q2words = remove_stopwords(row['question2'].split('|'), stops)

    if len(q1words) == 0 or len(q2words) == 0 : return 0

    shared_words_in_q1 = list_filter(q1words, lambda x: x in q2words)
    shared_words_in_q2 = list_filter(q2words, lambda x: x in q1words)
    
    return (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

def jaccard(row):
    wic = set(row['question1'].split('|')).intersection(set(row['question2'].split('|')))
    uw = set(row['question1'].split('|')).union(row['question2'].split('|'))
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1'].split('|')).intersection(set(row['question2'].split('|'))))

def total_unique_words(row):
    return len(set(row['question1'].split('|')).union(row['question2'].split('|')))

def total_unq_words_stop(row, stops={}):
    words = set(row['question1'].split('|')).union(row['question2'].split('|'))
    return len(remove_stopwords(words, stops))

def wc_diff(row):
    return abs(len(row['question1'].split('|')) - len(row['question2'].split('|')))

def calc_ratio(l1, l2):
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_ratio(row):
    l1 = len(row['question1'].split('|'))*1.0 
    l2 = len(row['question2'].split('|'))
    return calc_ratio(l1, l2)

def wc_diff_unique(row):
    return abs(len(set(row['question1'].split('|'))) - len(set(row['question2'].split('|'))))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'].split('|'))) * 1.0
    l2 = len(set(row['question2'].split('|')))
    return calc_ratio(l1, l2)

def wc_diff_unique_stop(row, stops={}):
    word1 = remove_stopwords(row['question1'].split('|'), stops)
    word2 = remove_stopwords(row['question2'].split('|'), stops)
    return abs(len(set(word1)) - len(set(word2)))

def wc_ratio_unique_stop(row, stops={}):
    l1 = len(set(remove_stopwords(row['question1'].split('|'), stops)))*1.0
    l2 = len(set(remove_stopwords(row['question2'].split('|'), stops)))
    return calc_ratio(l1, l2)

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'].split('|')[0] == row['question2'].split('|')[0])

def char_diff(row):
    return abs(len(''.join(row['question1'].split('|'))) - len(''.join(row['question2'].split('|'))))

def char_ratio(row):
    l1 = len(''.join(row['question1'].split('|'))) 
    l2 = len(''.join(row['question2'].split('|')))
    return calc_ratio(l1, l2)

def char_diff_unique_stop(row, stops={}):
    word1 = remove_stopwords(set(row['question1'].split('|')), stops)
    word2 = remove_stopwords(set(row['question2'].split('|')), stops)
    l1 = len(''.join(word1))
    l2 = len(''.join(word2))
    return abs(l1 - l2)

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)

def tfidf_word_match_share_stops(row, weights={}, stops={}):
    q1words = remove_stopwords(set(row['question1'].split('|')), stops)
    q2words = remove_stopwords(set(row['question2'].split('|')), stops)
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words if w in q2words] + [weights.get(w, 0) for w in q2words if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights={}, stops={}):
    q1words = remove_stopwords(set(row['question1'].split('|')), stops)
    q2words = remove_stopwords(set(row['question2'].split('|')), stops)
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words if w in q2words] + [weights.get(w, 0) for w in q2words if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1) #1

    f = functools.partial(tfidf_word_match_share, weights=weights, stops=stops)
    X['tfidf_wm'] = data.apply(f, axis=1) #2

    f = functools.partial(tfidf_word_match_share_stops, weights=weights, stops=stops)
    X['tfidf_wm_stops'] = data.apply(f, axis=1) #3

    X['jaccard'] = data.apply(jaccard, axis=1) #4
    X['wc_diff'] = data.apply(wc_diff, axis=1) #5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1) #6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1) #7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1) #8

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    X['wc_diff_unq_stop'] = data.apply(f, axis=1) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1) #10

    X['same_start'] = data.apply(same_start_word, axis=1) #11
    X['char_diff'] = data.apply(char_diff, axis=1) #12

    f = functools.partial(char_diff_unique_stop, stops=stops)
    X['char_diff_unq_stop'] = data.apply(f, axis=1) #13

#     X['common_words'] = data.apply(common_words, axis=1)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1)  #15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1)  #16
    
    X['char_ratio'] = data.apply(char_ratio, axis=1) #17    

    return X

def read_data(path):
    l1,l2,l3 = [],[],[]
    f = codecs.open(path, encoding="utf-8")
    for line in f:
        values = line.strip().split('\t')
        l1.append(values[0])
        l2.append(values[1])
        l3.append(int(float(values[2])))
    return l1,l2,l3

def ceate_feature_map(file_name,features):
    with open(file_name, 'w') as outfile:
        for i, feat in enumerate(features):
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))        
            #feature type, use i for indicator and q for quantity  
