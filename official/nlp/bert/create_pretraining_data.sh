#!/bin/sh

export PYTHONPATH="$PYTHONPATH:/home/shutt/repos/models"

python create_pretraining_data.py \
  --input_file=./sessions_med.txt \
  --output_file=/home/shutt/data/bert_model_and_data/data/sessions_med.tf \
  --output_pkl=/home/shutt/data/bert_model_and_data/data/sessions_med.pkl \
  --vocab_file=./vocab.txt \
  --do_lower_case=True \
  --max_seq_length=12 \
  --max_predictions_per_seq=5 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5