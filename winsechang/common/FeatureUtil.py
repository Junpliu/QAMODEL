# /usr/bin/python
# -*- coding:utf-8 -*-

"""
    Create by winsechang/2019.06.17
    Extract some features for sentence
"""

import sys
import pandas as pd
from common_function import *


class FeatureUtil:
    def __init__(self, q1_vec, q2_vec, stopword={}):
        self.__stop_word = stopword
        self.__q1 = q1_vec
        self.__q2 = q2_vec
        self.__q1_ust = remove_stopword(set(q1_vec), stopword)
        self.__q2_ust = remove_stopword(set(q2_vec), stopword)
        self.__c1 = "".join(q1_vec)
        self.__c2 = "".join(q2_vec)
        self.__c1_ust = remove_stopword(set(self.__c1), stopword)
        self.__c2_ust = remove_stopword(set(self.__c2), stopword)

    @property
    def jaccard(self):
        coml = self.common_words
        unil = self.union_words if self.union_words != 0 else 1
        return coml / unil

    @property
    def common_words(self):
        return len(set(self.__q1).intersection(set(self.__q2)))

    @property
    def union_words(self):
        return len(set(self.__q1).union(self.__q2))

    @property
    def total_unique_words(self):
        return len(set(self.__q1).union(self.__q2))

    @property
    def total_unq_words_stop(self):
        return len([x for x in set(self.__q1).union(self.__q2) if x not in self.__stop_word])

    @property
    def wc_diff(self):
        return abs(len(self.__q1) - len(self.__q2))

    @property
    def wc_diff_unique(self):
        return abs(len(set(self.__q1)) - len(set(self.__q2)))

    @property
    def wc_diff_unique_stop(self):
        return abs(len(self.__q1_ust) - len(self.__q2_ust))

    def calc_ratio(self, l1, l2):
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    @property
    def wc_ratio(self):
        l1 = len(self.__q1) * 1.0
        l2 = len(self.__q2)
        return self.calc_ratio(l1, l2)

    @property
    def wc_ratio_unique(self):
        l1 = len(set(self.__q1)) * 1.0
        l2 = len(set(self.__q2))
        return self.calc_ratio(l1, l2)

    @property
    def wc_ratio_unique_stop(self):
        l1 = len(self.__q1_ust) * 1.0
        l2 = len(self.__q2_ust)
        return self.calc_ratio(l1, l2)

    @property
    def same_start_word(self):
        return int(self.__q1[0] == self.__q2[0])

    @property
    def char_diff(self):
        return abs(len(self.__c1) - len(self.__c2))

    @property
    def char_ratio(self):
        l1 = len(self.__c1)
        l2 = len(self.__c2)
        return self.calc_ratio(l1, l2)

    @property
    def char_diff_unique_stop(self):
        return abs(len(self.__c1_ust) - len(self.__c2_ust))

    @property
    def get_weight(self, count, eps=10000, min_count=2):
        if count < min_count:
            return 0.0
        else:
            return 1.0 / (count + eps)

    @property  # TODO
    def tfidf_word_match_share_stops(self, weights=None):
        q1words = {}
        q2words = {}
        for word in self.__q1:
            if word not in stops:
                q1words[word] = 1
        for word in self.__q2:
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                        q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    @property
    def tfidf_word_match_share(self, weights=None):
        q1words = {}
        q2words = {}
        for word in self.__q1:
            q1words[word] = 1
        for word in self.__q2:
            q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                        q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R


if __name__ == "__main__":
    q1 = "天空|为什么|是|蓝色|的"
    q2 = "什么|原因|导致|天空|是|蓝色|的"
    FU = FeatureUtil(q1.split("|"), q2.split("|"), {"的"})

    print("q1 is : " + q1)
    print("q2 is : " + q2)
    print("jaccard distance: " + str(FU.jaccard))
    print("common_words: " + str(FU.common_words))
    print("total_unique_words: " + str(FU.total_unique_words))
    print("total_unq_words_stop: " + str(FU.total_unq_words_stop))
    print("wc_diff:" + str(FU.wc_diff))
    print("wc_diff_unique:" + str(FU.wc_diff_unique))
    print("wc_diff_unique_stop:" + str(FU.wc_diff_unique_stop))
    print("wc_ratio:" + str(FU.wc_ratio))
    print("wc_ratio_unique:" + str(FU.wc_ratio_unique))
    print("wc_ratio_unique_stop:" + str(FU.wc_ratio_unique_stop))
    print("same_start_word:" + str(FU.same_start_word))
    print("char_diff:" + str(FU.char_diff))
    print("char_ratio:" + str(FU.char_ratio))
    print("char_diff_unique_stop:" + str(FU.char_diff_unique_stop))
    # print (":" + str(FU.))

"""
def wc_ratio(row):
    l1 = len(self.__q1)*1.0 
    l2 = len(self.__q2)
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(self.__q1)) - len(set(self.__q2)))

def wc_ratio_unique(row):
    l1 = len(set(self.__q1)) * 1.0
    l2 = len(set(self.__q2))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(self.__q1) if x not in stops]) - len([x for x in set(self.__q2) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(self.__q1) if x not in stops])*1.0 
    l2 = len([x for x in set(self.__q2) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(self.__q1[0] == self.__q2[0])

def char_diff(row):
    return abs(len(''.join(self.__q1)) - len(''.join(self.__q2)))

def char_ratio(row):
    l1 = len(''.join(self.__q1)) 
    l2 = len(''.join(self.__q2))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(self.__q1) if x not in stops])) - len(''.join([x for x in set(self.__q2) if x not in stops])))

def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in self.__q1:
        if word not in stops:
            q1words[word] = 1
    for word in self.__q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in self.__q1:
        q1words[word] = 1
    for word in self.__q2:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1) #1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1) #2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
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

print("Starting to read training samples...")
train_texts_1 = [] 
train_texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:   # read as Unicode
    for line in f:
        values = line.strip().split('\t')
        train_texts_1.append(values[0])
        train_texts_2.append(values[1])
        labels.append(int(float(values[2])))

val_texts_1 = [] 
val_texts_2 = []
val_labels = []
with codecs.open(VAL_DATA_FILE, encoding='utf-8') as f:   # read as Unicode
    for line in f:
        values = line.strip().split('\t')
        val_texts_1.append(values[0])
        val_texts_2.append(values[1])
        val_labels.append(int(float(values[2])))

test_texts_1 = [] 
test_texts_2 = []
test_labels = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:   # read as Unicode
    for line in f:
        values = line.strip().split('\t')
        test_texts_1.append(values[0])
        test_texts_2.append(values[1])
        test_labels.append(int(float(values[2])))
print("Finish reading training samples !")

train_orig =  pd.DataFrame({"question1": train_texts_1, "question2": train_texts_2})
val_orig =  pd.DataFrame({"question1": val_texts_1, "question2": val_texts_2})
test_orig =  pd.DataFrame({"question1": test_texts_1, "question2": test_texts_2})

## basic features
train_cp = train_orig.copy()
val_cp = val_orig.copy()
test_cp = test_orig.copy()
x_train_basic = get_basic_feat(train_cp)
x_valid_basic = get_basic_feat(val_cp)
x_test_basic = get_basic_feat(test_cp)

## magic features
total_words = []
ques = pd.concat([train_orig, val_orig], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    total_words += ques.question1[i].split('|')
    q_dict[ques.question2[i]].add(ques.question1[i])
    total_words += ques.question2[i].split('|')
pk.dump(total_words, open('total_words', 'wb'))

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

#x_train_magic = pd.DataFrame()
#x_valid_magic = pd.DataFrame()
#x_train_magic['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1)
#x_train_magic['q1_freq'] = train_orig.apply(q1_freq, axis=1)
#x_train_magic['q2_freq'] = train_orig.apply(q2_freq, axis=1)

#x_valid_magic['q1_q2_intersect'] = val_orig.apply(q1_q2_intersect, axis=1)
#x_valid_magic['q1_freq'] = val_orig.apply(q1_freq, axis=1)
#x_valid_magic['q2_freq'] = val_orig.apply(q2_freq, axis=1)

## tfidf features
counts = Counter(total_words)
weights = {word: get_weight(count) for word, count in counts.items()}
x_train_more = build_features(train_orig, stop_words, weights)
x_valid_more = build_features(val_orig, stop_words, weights)
x_test_more = build_features(test_orig, stop_words, weights)
## save word freq to total_counts
r = open('total_counts', 'w')
for _word, _count in counts.items():
    r.write("%s\t%d\n"%(_word.encode('utf8'), _count))
r.close()

## combine all features
x_train = pd.concat((x_train_basic, x_train_more), axis=1)
x_valid = pd.concat((x_valid_basic, x_valid_more), axis=1)
x_test = pd.concat((x_test_basic, x_test_more), axis=1)
x_train.columns = [str(i) for i in range(x_train.shape[1])]
x_valid.columns = [str(i) for i in range(x_valid.shape[1])]
x_test.columns = [str(i) for i in range(x_test.shape[1])]
print(x_train)

## train models
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['error', 'logloss']
params['eta'] = 0.08
params['max_depth'] = 6

d_train = xgb.DMatrix(x_train, label=labels)
d_valid = xgb.DMatrix(x_valid, label=val_labels)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model_path = 'model'
#bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50)
bst.dump_model(model_path + '.dump')
bst.save_model(model_path)

## make the submission
p_test = bst.predict(xgb.DMatrix(x_test))
df_sub = pd.DataFrame({'user_query':test_texts_1, 'candidate_query':test_texts_2, 'label':test_labels, 'score':p_test.ravel()})
df_sub.to_csv('data/partition/results_xgb.csv', index=False, columns=['user_query', 'candidate_query', 'label', 'score'], encoding='utf-8')

"""
