#!/bin/sh
python run_pretraining.py \
  --input_files=$HOME/data/bert_model_and_data/data/sessions_small.tf \
  --model_export_path=$HOME/data/bert_model_and_data/model/pretraining_output \
  --bert_config_file=./bert_config_small.json \
  --train_batch_size=16 \
  --max_seq_length=12 \
  --max_predictions_per_seq=5 \
  --num_steps_per_epoch=1000 \
  --warmup_steps=10000 \
  --learning_rate=2e-5