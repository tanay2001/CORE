#!/bin/bash

model=

# mnli
export TASK_NAME=mnli
NUM_GPU=3
OUTPUT_DIR=
python3 test_glue.py \
  --model_name_or_path $model \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $OUTPUT_DIR \

# snli
export TASK_NAME=snli
NUM_GPU=3
OUTPUT_DIR=

python3 test_glue.py \
  --model_name_or_path $model \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $OUTPUT_DIR \


# diago 
export TASK_NAME=diago
NUM_GPU=3
TEST_FILE=
OUTPUT_DIR=
python3 test_glue.py \
  --model_name_or_path $model \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $OUTPUT_DIR \
  --test_file $TEST_FILE \
  --train_file $TEST_FILE \

# adv
export TASK_NAME=adv
NUM_GPU=3
TEST_FILE=
OUTPUT_DIR=
python3 test_glue.py \
  --model_name_or_path $model \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $OUTPUT_DIR \
  --test_file $TEST_FILE \
  --train_file $TEST_FILE

# wanli
export TASK_NAME=wanli
NUM_GPU=3
OUTPUT_DIR=
TEST_FILE=
python3 test_glue.py \
  --model_name_or_path $model \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $OUTPUT_DIR \
  --test_file $TEST_FILE \
  --train_file $TEST_FILE \

# anli
export TASK_NAME=anli
NUM_GPU=3
CAT=v3
OUTPUT_DIR=

python3 test_glue.py \
  --model_name_or_path $model \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 64 \
  --output_dir $OUTPUT_DIR/v3 \
