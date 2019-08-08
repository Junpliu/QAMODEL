#!/usr/bin/env bash
export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190806/raw
export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/data
export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/model
export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/tmp
export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/log
nohup python xgb_predict.py \
    --test_data_file $RAW_DATA_PATH/test.txt \
    --x_test_file $DATA_PATH/x_test.csv \
    --y_test_file $DATA_PATH/y_test.csv \
    --dtest_file $DATA_PATH/dtest.buffer \
    --pred_data_file $TMP_PATH/results_xgb.csv_tmp_ \
    --model_path $MODEL_PATH/model \
    --use_scwlstm False \
    --ntree_limit 0 \
    > $LOG_DIR/predict_with_scwlstm.log 2>&1 &


#export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190806/raw
#export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/data_with_scwlstm
#export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/model_with_scwlstm
#export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/tmp_with_scwlstm
#export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/log
#nohup python xgb_predict.py \
#    --test_data_file $RAW_DATA_PATH/test.txt \
#    --x_test_file $DATA_PATH/x_test.csv \
#    --y_test_file $DATA_PATH/y_test.csv \
#    --dtest_file $DATA_PATH/dtest.buffer \
#    --pred_data_file $TMP_PATH/results_xgb.csv_tmp_ \
#    --model_path $MODEL_PATH/model \
#    --use_scwlstm True \
#    --ntree_limit None \
#    > $LOG_DIR/predict_with_scwlstm.log 2>&1 &