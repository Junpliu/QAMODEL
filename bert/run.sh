#!/usr/bin/env bash
export BERT_BASE_DIR=/ceph/qbkg/aitingliu/qq/chinese_L-12_H-768_A-12
export MY_DATASET=/ceph/qbkg/aitingliu/qq/data/20190809/for_bert
python bert/run_classifier.py \
  --task_name=qq \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=40 \
  --train_batch_size=128 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/ceph/qbkg/aitingliu/qq/bert_model/20190809/