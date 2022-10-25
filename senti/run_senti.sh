OUTPUT_DIR=
TRAIN_FILE=
DEV_FILE=
NUM_GPU=3
CAT=sentiment
python3 run_senti.py \
  --do_train \
  --model_name_or_path microsoft/deberta-base \
  --do_eval \
  --save_total_limit 2 \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --save_steps 400 \
  --load_best_model_at_end True \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 5 \
  --output_dir $OUTPUT_DIR \
  --train_file $TRAIN_FILE \
  --validation_file $DEV_FILE