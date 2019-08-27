#!/usr/bin/env bash
#export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190806/raw
#export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/data
#export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/xgbmodel
#export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/xgbmodel/tmp
#export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/xgbmodel/log
#nohup python xgb_predict.py \
#    --test_data_file $RAW_DATA_PATH/test.txt \
#    --x_test_file $DATA_PATH/x_test.csv \
#    --y_test_file $DATA_PATH/y_test.csv \
#    --dtest_file $DATA_PATH/dtest.buffer \
#    --pred_data_file $TMP_PATH/results_xgb.csv_tmp_ \
#    --model_path $MODEL_PATH/model.dump \
#    --no_use_scwlstm \
#    --ntree_limit 0 \
#    > $LOG_DIR/predict.log 2>&1 &


#export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190806/raw
#export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/data_with_scwlstm
#export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/model_with_scwlstm
#export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/tmp_with_scwlstm
#export SCWLSTM_MODEL_PATH=/ceph/qbkg2/aitingliu/qq/src/model/20190806/SCWLSTM
#export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/20190806/log
#nohup python xgb_predict.py \
#    --test_data_file $RAW_DATA_PATH/test.txt \
#    --test_scwlstm_pred_file $SCWLSTM_MODEL_PATH/best_eval_loss/output_test
#    --use_scwlstm \
#    --x_test_file $DATA_PATH/x_test.csv \
#    --y_test_file $DATA_PATH/y_test.csv \
#    --dtest_file $DATA_PATH/dtest.buffer \
#    --pred_data_file $TMP_PATH/results_xgb.csv_tmp_ \
#    --word_counts_file $TMP_PATH/word_counts \
#    --model_path $MODEL_PATH/model \
#    --ntree_limit 0 \
#    > $LOG_DIR/predict_with_scwlstm.log 2>&1 &



export RAW_DATA_PATH=/ceph/qbkg2/aitingliu/qq/data/20190726/raw
export DATA_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190726/data_v2
export MODEL_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190726/model_v2
export TMP_PATH=/ceph/qbkg2/aitingliu/qq/XGBoost/20190726/tmp_v2
export SCWLSTM_MODEL_PATH=/ceph/qbkg2/aitingliu/qq/src/model/20190726/SCWLSTM
export LOG_DIR=/ceph/qbkg2/aitingliu/qq/XGBoost/20190726/log
nohup python xgb_predict.py \
    --first_predict \
    --test_data_file $RAW_DATA_PATH/merge_20190816_check2_seg.txt \
    --test_scwlstm_pred_file None \
    --no_use_scwlstm \
    --x_test_file $DATA_PATH/x_merge_20190816_check2_seg.csv \
    --y_test_file $DATA_PATH/y_merge_20190816_check2_seg.csv \
    --dtest_file $DATA_PATH/dtest_merge_20190816_check2_seg.buffer \
    --pred_data_file $TMP_PATH/results_xgb.csv_tmp_ \
    --word_counts_file $TMP_PATH/word_counts \
    --model_path $MODEL_PATH/model \
    --ntree_limit 0 \
    > $LOG_DIR/predict_v2.log 2>&1 &

