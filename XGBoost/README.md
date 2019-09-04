# XGBoost Model

## Train

参考 `run.sh`

## Predict

参考 `predict.sh`

## Evaluate

evaluate.py

## Params

max_depth:树最大深度 
eta:迭代步长 
silent：是否打印额外信息 
objective：训练目标 
booster：树类型 
gamma：子树生成最小loss 
tree_method：树生成类型 
lambda：L2正则项，防止过拟合 
alpha：L1正则项，防止过拟合 
subsample:样本测样，防止过拟合 
colsample_bytree：建树特征抽样，防止过拟合 
colsample_bylevel：子树特征抽样，防止过拟合

## Refs

XGBoost parameters: https://xgboost.readthedocs.io/en/latest/parameter.html

XGBClassifier: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier

GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

## Result



## Update

@20190731：重构winsechang的代码



## TODO

20190802： 和景冬用同一个测试集，扩增数据集后的训练集不能包括之前测试集里的样本