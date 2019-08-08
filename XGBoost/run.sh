#!/usr/bin/env bash
#export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190806/raw
#export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/data
#export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/model
#export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/tmp
#export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/log
#nohup python xgb_train.py \
#        --first_train \
#        --model_path $MODEL_PATH/model \
#        --train_data_file $RAW_DATA_PATH/train.txt \
#        --valid_data_file $RAW_DATA_PATH/dev.txt \
#        --test_data_file $RAW_DATA_PATH/test.txt \
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
#        --eta 0.08 \
#        > $LOG_DIR/train.log 2>&1 &

export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190806/raw
export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/data_with_scwlstm
export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/model_with_scwlstm
export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/tmp_with_scwlstm
export SCWLSTM_MODEL_PATH=/ceph/qbkg2/aitingliu/qq/src/model/SCWLSTM/20190806
export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/log
nohup python xgb_train.py \
        --first_train \
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
        --train_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_train \
        --valid_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_dev \
        --test_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_test \
        --word_counts_file $TMP_PATH/word_counts \
        --feature_map_file $TMP_PATH/feature_map \
        --pred_data_file $TMP_PATH/results_xgb.csv \
        --use_scwlstm \
        --num_boost_round 2000 \
        --eta 0.08 \
        > $LOG_DIR/train_with_scwlstm.log 2>&1 &


