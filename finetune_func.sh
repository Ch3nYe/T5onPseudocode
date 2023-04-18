#!/bin/zsh
NUM_EPOCH=200
WANDB_ENTITY='xiong'
TASK="finetune-func-name_Nero"
LR=5e-4
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
NUM_PROCESS=5
MODEL_NAME="record/t2t_ckpt_2/5"
MODEL_TYPE="t5"
T2T_BATCH_SIZE_DEVICE=32
EVAL_BATCH_SIZE_DEVICE=32
WANDB_PROJECT="FINTUNE_FUNC_NAME_DEEPSPEED"
SHUFFLE=True
SEED=340712
EVAL_MAX_LENGTH=10
NUM_WORKERS=8
OUTPUT_DIR="record/fine_tune_Nero"
IGNORE_MINIMUM_LENGTH=19
USE_DEEPSPEED=True
RESUME_FROM_CHECKPOINT=""
GRADIENT_ACCUMULATION_STEPS=1
WITH_TRACKING=True
MAX_TGT_LENGTH=16
# EVAL_SET_SPLIT_TYPE="function"
DROP_LAST=False
accelerate-launch --main_process_port=28713 --num_processes=$NUM_PROCESS --dynamo_backend=no finetune_func_name.py --model_name=$MODEL_NAME --model_type=$MODEL_TYPE --debug --num_train_epochs=$NUM_EPOCH  --per_device_train_batch_size=$T2T_BATCH_SIZE_DEVICE --output_dir=$OUTPUT_DIR --num_worker=$NUM_WORKERS  --learning_rate=$LR   --wandb_logging_steps=500 --shuffle=$SHUFFLE --wandb_project=$WANDB_PROJECT --wandb_entity=$WANDB_ENTITY --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS --manual_seed=$SEED  --val_max_target_length=$EVAL_MAX_LENGTH --ignore_minimum_length=$IGNORE_MINIMUM_LENGTH  --per_device_eval_batch_size=$EVAL_BATCH_SIZE_DEVICE --with_tracking=$WITH_TRACKING --max_tgt_length=$MAX_TGT_LENGTH --eval_set_split_type=$EVAL_SET_SPLIT_TYPE --drop_last=$DROP_LAST