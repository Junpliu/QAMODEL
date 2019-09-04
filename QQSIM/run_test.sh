#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 nohup python3 main_apcnn.py --train=false --test=true > log/test_apcnn 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 main_aplstm.py --train=false --test=true > log/test_aplstm 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 main_scwlstm.py --train=false --test=true > log/test_scwlstm 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 main_fasttext.py --train=false --test=true > log/test_fasttext 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 main_textcnn.py --train=false --test=true > log/test_textcnn 2>&1 &