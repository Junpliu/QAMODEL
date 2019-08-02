# XGBoost Model

## Train

```bash
export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190726/raw/
export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/
export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/model/with_scwlstm/
export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/
nohup python xgb_train.py \
        --model_path $MODEL_PATH/model \
        --train_data_file $RAW_DATA_PATH/train.txt \
        --valid_data_file $RAW_DATA_PATH/dev.txt \
        --test_data_file $RAW_DATA_PATH/test.txt \
        --x_train_file $DATA_PATH/x_train.csv \
        --x_valid_file $DATA_PATH/x_valid.csv \
        --x_test_file $DATA_PATH/x_test.csv \
        --y_train_file $DATA_PATH/y_train.csv \
        --y_valid_file $DATA_PATH/y_valid.csv \
        --y_test_file $DATA_PATH/y_test.csv \
        --dtrain_file $DATA_PATH/dtrain.buffer \
        --dvalid_file $DATA_PATH/dvalid.buffer \
        --dtest_file $DATA_PATH/dtest.buffer \
        --word_counts_file $TMP_PATH/word_counts \
        --feature_map_file $TMP_PATH/feature_map \
        --pred_data_file $TMP_PATH/results_xgb.csv \
        --use_scwlstm True \
        --eta 0.1 \
        --colsample_bytree 0.7 \
        > train_with_scwlstm.log 2>&1 &
```

## Predict
```bash
export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190726/raw/
export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm/
export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/model/with_scwlstm/
export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm/
nohup python xgb_predict.py \
    --test_data_file $RAW_DATA_PATH/test.txt \
    --x_test_file $DATA_PATH/x_test.csv \
    --y_test_file $DATA_PATH/y_test.csv \
    --dtest_file $DATA_PATH/dtest.buffer \
    --pred_data_file $TMP_PATH/results_xgb.csv_tmp_ \
    --model_path $MODEL_PATH/model \
    --use_scwlstm True \
    --ntree_limit None \
    > predict_with_scwlstm.log 2>&1 &
```

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