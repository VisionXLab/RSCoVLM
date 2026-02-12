#!/bin/bash

NNODES=${NNODES:-4}
PER_RANK_BATCH_SIZE=${PER_RANK_BATCH_SIZE:-1}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-32}

MIN_PIXELS=${MIN_PIXELS:-64}  # 64->224, 1296->1008, 16384->3584, 20736->4032
MAX_PIXELS=${MAX_PIXELS:-1296}
LEARNING_RATE=${LEARNING_RATE:-0.000002}
PROB_RANDOM_RESIZE=${PROB_RANDOM_RESIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-6144}
ZERO_STAGE=${ZERO_STAGE:-1}

NGPUS=$((NNODES * 8))
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_RANK_BATCH_SIZE * NGPUS)))

RUN_NAME=${RUN_NAME:-"qwen2-5-vl-7b-ins_sft0819_tune-all-plain-mode-keep-empty-gt-prob-random-resize-${PROB_RANDOM_RESIZE}_pix${MIN_PIXELS}-${MAX_PIXELS}_lr${LEARNING_RATE}-b${PER_RANK_BATCH_SIZE}x${NGPUS}GPUxga${GRAD_ACCUM_STEPS}_${MAX_LENGTH}len_noflatten-zero${ZERO_STAGE}"}

export TRITON_CACHE_DIR="/tmp/triton_rscovlm/"

set -x

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc-per-node=8 --nnodes=${NNODES} \
    -m rscovlm.training.train \
    --deepspeed rscovlm/config/internvl/zero_stage${ZERO_STAGE}_config.json \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --datasets rscovlm/config/data.json \
    --min_pixels $((MIN_PIXELS * 28 * 28)) \
    --max_pixels $((MAX_PIXELS * 28 * 28)) \
    --prob_proxy_prompt 0 \
    --prob_plain_text_prompt 1 \
    --keep_empty_gt True \
    --prob_random_resize ${PROB_RANDOM_RESIZE} \
    --lora_enable False \
    --vision_lora False \
    --freeze_vision False \
    --freeze_language False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --use_liger_kernel True \
    --num_train_epochs 1 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size $PER_RANK_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --save_strategy "steps" \
    --save_steps 5000 \
    --run_name $RUN_NAME \
    --output_dir ./output/$RUN_NAME \
    --max_length $MAX_LENGTH
