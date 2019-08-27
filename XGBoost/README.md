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

### 20190802

| 模型            | 指标            | AUC    | ACC    | PRE    | REC    | F1     | 阈值 | 备注 |
| --------------- | --------------- | ------ | ------ | ------ | ------ | ------ | ---- | ---- |
| xgb(winsechang) | QQ Pair级别指标 |        | 0.6651 | 0.8823 | 0.1584 | 0.2686 | 0.9  |      |
| xgb(winsechang) | Query级别指标   |        | 0.5651 | 0.8923 | 0.2836 | 0.4304 | 0.9  |      |
| xgb             | QQ Pair级别指标 | 0.9200 | 0.6718 | 0.9854 | 0.1569 | 0.2707 | 0.87 |      |
| xgb             | Query级别指标   |        | 0.5341 | 0.9825 | 0.2894 | 0.4471 | 0.87 |      |
| xgb_check       | QQ Pair级别指标 |        |        |        |        |        |      |      |
| xgb_check       | Query级别指标   |        |        |        |        |        |      |      |



## Update

@20190731：重构winsechang的代码



## TODO

20190802： 和景冬用同一个测试集，扩增数据集后的训练集不能包括之前测试集里的样本