export TRITON_CACHE_DIR="/tmp/triton_rscovlm/"

torchrun ${distributed_args} -m rscovlm.eval.run \
    --benchmarks dota_test512 \
    --use_vllm \
    --model_ckpt_path ${@:1}

torchrun ${distributed_args} -m rscovlm.eval.run \
    --benchmarks \
    lrsvqa lrsvqa_visual_cot \
    mme_realworld_remote_sensing mme_realworld_remote_sensing_visual_cot \
    --batch_size 1 \
    --model_ckpt_path ${@:1}

torchrun ${distributed_args} -m rscovlm.eval.run \
    --benchmarks \
    teochatlas_geochat \
    cls_aid \
    vrsbench_vqa \
    vhm_scene_cls vhm_rsvqa \
    lrsvqa lrsvqa_visual_cot \
    mme_realworld_remote_sensing mme_realworld_remote_sensing_visual_cot \
    --max_max_length 2048 \
    --batch_size 128 \
    --model_ckpt_path ${@:1}

torchrun $distributed_args \
    -m rscovlm.eval.run \
    --benchmarks geoground_224 geoground_336 geoground_448 geoground_1008 geoground \
    --use_vllm \
    --model_ckpt_path \
    output/qwen2-5-vl-7b-ins_sft0819_tune-all-plain-mode-keep-empty-gt-prob-random-resize-1_pix64-1296_lr0.000002-b1x32GPUxga1_6144len_noflatten-zero1