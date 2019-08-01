#!/usr/bin/env bash
nohup python xgb_train.py \
        --model_path model/with_scwlstm1/model \
        --word_counts_file tmp/with_scwlstm1/word_counts \
        --feature_map_file tmp/with_scwlstm1/feature_map \
        --pred_data_file tmp/with_scwlstm1/results_xgb.csv \
        --use_scwlstm True \
        --eta 0.1 \
        --colsample_bytree 0.7 \
        --colsample_bylevel 0.5 \
        > train_with_scwlstm1.log 2>&1 &



