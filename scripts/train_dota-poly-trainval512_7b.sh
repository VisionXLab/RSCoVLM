#!/bin/bash

NNODES=${NNODES:-2}
PER_RANK_BATCH_SIZE=${PER_RANK_BATCH_SIZE:-2}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-32}

MIN_PIXELS=${MIN_PIXELS:-256}  # 256->448, 1296->1008, 16384->3584, 20736->4032
MAX_PIXELS=${MAX_PIXELS:-1296}
LEARNING_RATE=${LEARNING_RATE:-0.000002}
EPOCHS=${EPOCHS:-10}
PROB_RANDOM_RESIZE=${PROB_RANDOM_RESIZE:-0.0}
MAX_LENGTH=${MAX_LENGTH:-6144}
ZERO_STAGE=${ZERO_STAGE:-1}

NGPUS=$((NNODES * 8))
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_RANK_BATCH_SIZE * NGPUS)))

RUN_NAME=${RUN_NAME:-"qwen2-5-vl-7b-ins_dota-poly-trainval512_tune-all-plain-mode-keep-empty-gt-prob-random-resize-${PROB_RANDOM_RESIZE}_pix${MIN_PIXELS}-${MAX_PIXELS}_lr${LEARNING_RATE}-${EPOCHS}epochs-b${PER_RANK_BATCH_SIZE}x${NGPUS}GPUxga${GRAD_ACCUM_STEPS}_${MAX_LENGTH}len_noflatten-zero${ZERO_STAGE}"}

export TRITON_CACHE_DIR="/tmp/triton_lqy/"

set -x

ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes=${NNODES} --nproc-per-node=8 \
    --nproc-per-node=8 --nnodes=${NNODES} \
    -m rscovlm.training.train \
    --deepspeed rscovlm/config/internvl/zero_stage${ZERO_STAGE}_config.json \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --datasets dota_poly_trainval512 \
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
    --num_train_epochs ${EPOCHS} \
    --dataloader_num_workers 8 \
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
    --save_steps 1250 \
    --save_total_limit 2 \
    --run_name $RUN_NAME \
    --output_dir ./output/$RUN_NAME \
    --max_length $MAX_LENGTH
