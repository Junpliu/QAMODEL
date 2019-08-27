#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: changjingdong
Date: 20190614
Desc: xgboost model to predict similar questions

Update: aitingliu, 20190731
"""
import os
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
plt.switch_backend('agg')


parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--model_path", type=str, default="20190809/model_with_scwlstm7/model")
parser.add_argument("--max_num_features", type=int, default=30)
parser.add_argument("--fig_path", type=str, default="20190809/model_with_scwlstm7/fig/")
args = parser.parse_args()


bst = xgb.Booster(model_file=args.model_path)  # init model
fig, ax = plt.subplots()
xgb.plot_importance(bst, ax=ax, max_num_features=args.max_num_features)
fig_name = os.path.join(args.fig_path, "feature_{}.png".format(args.max_num_features))
plt.savefig(fig_name)
