#/usr/bin/python
# -*- coding:utf-8 -*-
'''
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions
'''
import sys
import codecs
from collections import defaultdict
from collections import Counter
import pandas as pd
import numpy as np
import functools
import xgboost as xgb
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import pickle as pk
import os
from data_helper import *
sys.path.insert(0, '../common')
from commen_function import *
from xgboost import plot_importance
#
x_test = xgb.DMatrix('test.buffer_1000')

#
bst = xgb.Booster({'nthread': 1})  # init model
bst.load_model('model/model')  # load data

#p_test = bst.predict(x_test)
#
#for x in p_test:
#    print  (x)
#
##

plot_importance(bst)
plt.show()
