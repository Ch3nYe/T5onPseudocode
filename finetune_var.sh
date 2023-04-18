#!/bin/zsh
NUM_EPOCH=5
WANDB_ENTITY='xiong'
TASK="finetune-var-name"
LR=5e-4
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7,8,9
NUM_PROCESS=8
MODEL_NAME="record/fine_tune_var_name_ckpt/2"
MODEL_TYPE="t5"
T2T_BATCH_SIZE_DEVICE=20
EVAL_BATCH_SIZE_DEVICE=16
WANDB_PROJECT="FINTUNE_VAR_NAME_DEEPSPEED"
SHUFFLE=True
SEED=312407
EVAL_MAX_LENGTH=256
NUM_WORKERS=8
OUTPUT_DIR="record/fine_tune_var_name_ckpt_2"
IGNORE_MINIMUM_LENGTH=19
USE_DEEPSPEED=True
RESUME_FROM_CHECKPOINT=""
GRADIENT_ACCUMULATION_STEPS=1
WITH_TRACKING=True
MAX_TGT_LENGTH=256
CHECKPOINTING_STEPS=30000
EVAL_ONLY=False
accelerate-launch --main_process_port=22119 --num_processes=$NUM_PROCESS --dynamo_backend=no finetune_var_name.py --model_name=$MODEL_NAME --model_type=$MODEL_TYPE --debug --num_train_epochs=$NUM_EPOCH  --per_device_train_batch_size=$T2T_BATCH_SIZE_DEVICE --output_dir=$OUTPUT_DIR --num_worker=$NUM_WORKERS  --learning_rate=$LR   --wandb_logging_steps=500 --shuffle=$SHUFFLE --wandb_project=$WANDB_PROJECT --wandb_entity=$WANDB_ENTITY --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS --manual_seed=$SEED  --val_max_target_length=$EVAL_MAX_LENGTH --ignore_minimum_length=$IGNORE_MINIMUM_LENGTH  --per_device_eval_batch_size=$EVAL_BATCH_SIZE_DEVICE --with_tracking=$WITH_TRACKING --max_tgt_length=$MAX_TGT_LENGTH --checkpointing_steps=$CHECKPOINTING_STEPS --eval_only=$EVAL_ONLY