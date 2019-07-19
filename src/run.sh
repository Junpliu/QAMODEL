#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=3 nohup python3 main_apcnn.py > log/train_apcnn 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python3 main_aplstm.py > log/train_aplstm 2>&1 &
#CUDA_VISIBLE_DEVICES=5 nohup python3 main_scwlstm.py > log/train_scwlstm 2>&1 &
#CUDA_VISIBLE_DEVICES=6 nohup python3 main_fasttext.py > log/train_fasttext 2>&1 &
#CUDA_VISIBLE_DEVICES=7 nohup python3 main_textcnn.py > log/train_textcnn 2>&1 &
#python3 main_fasttext.py --train=false --test=true