#!/usr/bin/env bash
#nohup python xgb_predict.py \
#    --test_data_file /ceph/qbkg2/winsechang/MODEL/qq_simscore/src/XGBoost/data/partition/test.txt \
#    --pred_data_file tmp/results_xgb.csv_tmp_win_ \
#    > predict.log 2>&1 &

nohup python xgb_predict.py \
    --test_data_file ../../data/20190726/raw/test.txt \
    --pred_data_file tmp/with/scwlstm/results_xgb.csv_tmp_ \
    --model_path model/with_scwlstm/model \
    --use_scwlstm True \
    --ntree_limit 140 \
    > predict_ws.log 2>&1 &