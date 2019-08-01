# train model
python xgb.py

# evaluate model
python ../common/metric.py  $result_data  $thr

# predict 
...


# 参数
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
