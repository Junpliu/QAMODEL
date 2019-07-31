#!/usr/bin/env bash
nohup python xgb_train.py \
        --model_path model/with_scwlstm/model \
        --word_counts_file tmp/with_scwlstm/word_counts \
        --feature_map_file tmp/with_scwlstm/feature_map \
        --pred_data_file tmp/with_scwlstm/results_xgb.csv \
        --use_scwlstm True \
        > train_with_scwlstm.log 2>&1 &



