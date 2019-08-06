#!/usr/bin/env bash
#export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190726/raw
#export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/data
#export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/model
#export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/tmp
#nohup python xgb_train.py \
#        --model_path $MODEL_PATH/model \
#        --train_data_file $RAW_DATA_PATH/train_check.txt \
#        --valid_data_file $RAW_DATA_PATH/dev.txt \
#        --test_data_file $RAW_DATA_PATH/test_check.txt \
#        --x_train_file $DATA_PATH/x_train.csv \
#        --x_valid_file $DATA_PATH/x_valid.csv \
#        --x_test_file $DATA_PATH/x_test.csv \
#        --y_train_file $DATA_PATH/y_train.csv \
#        --y_valid_file $DATA_PATH/y_valid.csv \
#        --y_test_file $DATA_PATH/y_test.csv \
#        --dtrain_file $DATA_PATH/dtrain.buffer \
#        --dvalid_file $DATA_PATH/dvalid.buffer \
#        --dtest_file $DATA_PATH/dtest.buffer \
#        --word_counts_file $TMP_PATH/word_counts \
#        --feature_map_file $TMP_PATH/feature_map \
#        --pred_data_file $TMP_PATH/results_xgb.csv \
#        --no_use_scwlstm \
#        --eta 0.1 \
#        > train.log 2>&1 &
#
export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190726/raw
export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/data/with_scwlstm2
export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/model/with_scwlstm2
export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/tmp/with_scwlstm2
export SCWLSTM_MODEL_PATH=/ceph/qbkg2/aitingliu/qq/src/model/SCWLSTM
nohup python xgb_train.py \
        --model_path $MODEL_PATH/model \
        --train_data_file $RAW_DATA_PATH/train_check.txt \
        --valid_data_file $RAW_DATA_PATH/dev.txt \
        --test_data_file $RAW_DATA_PATH/test_check.txt \
        --x_train_file $DATA_PATH/x_train.csv \
        --x_valid_file $DATA_PATH/x_valid.csv \
        --x_test_file $DATA_PATH/x_test.csv \
        --y_train_file $DATA_PATH/y_train.csv \
        --y_valid_file $DATA_PATH/y_valid.csv \
        --y_test_file $DATA_PATH/y_test.csv \
        --dtrain_file $DATA_PATH/dtrain.buffer \
        --dvalid_file $DATA_PATH/dvalid.buffer \
        --dtest_file $DATA_PATH/dtest.buffer \
        --train_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_train_check \
        --valid_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_dev \
        --test_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_test_check \
        --word_counts_file $TMP_PATH/word_counts \
        --feature_map_file $TMP_PATH/feature_map \
        --pred_data_file $TMP_PATH/results_xgb.csv \
        --use_scwlstm \
        > train_with_scwlstm2.log 2>&1 &



