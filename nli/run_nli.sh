#!/bin/bash

export TASK_NAME=cf
OUTPUT_DIR=
TRAIN_FILE=
NUM_GPU=3
python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU run_glue.py \
  --model_name_or_path microsoft/deberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --save_total_limit 2 \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --save_steps 400 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT_DIR \
  --load_best_model_at_end True \
  --train_file $TRAIN_FILE \
