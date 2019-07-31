# /usr/bin/python
# -*- coding:utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions

Update: aitingliu, 20190731
"""
import xgboost as xgb
import matplotlib.pyplot as plt
plt.switch_backend('agg')

bst = xgb.Booster(model_file="./model/model")  # init model
fig, ax = plt.subplots()
xgb.plot_importance(bst, ax=ax, max_num_features=50)
plt.savefig("fig/feature50.png")
