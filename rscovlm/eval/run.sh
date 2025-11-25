#!/bin/bash

# BENCHMARKS=('vrsbench_caption' 'vrsbench_caption_1' 'vrsbench_caption_2' 'vrsbench_caption_4' 'vrsbench_caption_8' 'vrsbench_caption_16')
# BENCHMARKS=('cls_aid' 'cls_aid_4' 'cls_resisc' 'cls_resisc_4')
# BENCHMARKS=('cls_aid' 'cls_aid_unique_4' 'cls_resisc' 'cls_resisc_unique_4')
# BENCHMARKS=('vrsbench_vqa' 'vrsbench_vqa_4')
# ('cap_nwpu_caption' 'vrsbench_caption' 'cap_rsicd' 'cap_rsitmd' 'cap_sydney_caption' 'cap_ucm_caption')
# BENCHMARKS=('refcoco_val_4' 'refcocop_val_4' 'refcocog_val_4' 'dior_rsvg_val_4') 
BENCHMARKS=('refcoco_val')
# BENCHMARKS=('dior_rsvg_val_1')
# BENCHMARKS=('geochat_aid' 'geochat_ucmerced' 'geochat_grounding_description' 'geochat_region_captioning' 'geochat_hrben' 'geochat_lrben' 'geochat_referring')
# BENCHMARKS=('geochat_aid') 
# BENCHMARKS=('geochat_ucmerced')
# BENCHMARKS=('geochat_grounding_description')
# BENCHMARKS=('geochat_region_captioning')
# BENCHMARKS=('geochat_hrben')
# BENCHMARKS=('geochat_lrben')
# BENCHMARKS=('geochat_referring') 

# ("vrsbench_referring") # label有错

# geochat_grounding_description的回答不太合理：参考07558.jpg，最重要的chimney没有说。
# TODO: 问
# TODO:给vqa和caption加icl
# TODO:给refering也加icl，否则plain text mode只生成一个bbox

SAMPLE_NUM=256

# torchrun
PYTHONPATH="$PYTHONPATH:$(pwd)" \
torchrun --nproc_per_node=1 --master_port=29501 \
    run.py \
    --benchmarks "${BENCHMARKS[@]}" \
    --batch_size 32 \
    # --sample_num ${SAMPLE_NUM} \
    # --grouding_use_plain_text_mode \

