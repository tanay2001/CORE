#!/bin/bash

model=

# IMDB
dataset=imdb
DATASET_FILE=
python3 run_senti.py \
  --model_name_or_path $model \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $model/$dataset \
  --train_file $DATASET_FILE \
  --validation_file $DATASET_FILE

# SST2

dataset=sst2

python3 run_senti.py \
  --model_name_or_path $model \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $model/$dataset \
  --task_name sst2

# CONTRAST

dataset=cont
DATASET_FILE=
python3 run_senti.py \
  --model_name_or_path $model \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $model/$dataset \
  --train_file $DATASET_FILE \
  --validation_file $DATASET_FILE

# CAD
dataset=cad
DATASET_FILE=
python3 run_senti.py \
  --model_name_or_path $model \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $model/$dataset \
  --train_file $DATASET_FILE \
  --validation_file $DATASET_FILE

# Senti140
dataset=senti140
DATASET_FILE=
python3 run_senti.py \
  --model_name_or_path $model \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $model/$dataset \
  --train_file $DATASET_FILE \
  --validation_file $DATASET_FILE


# yelp_polarity
export TASK_NAME=yelp_polarity
dataset=yelp

python3 run_senti.py \
  --model_name_or_path $model \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $model/$dataset \
  --task_name $TASK_NAME \