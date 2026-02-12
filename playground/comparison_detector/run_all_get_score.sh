#!/bin/bash

# Create 4x2 grid layout (8 panes total)
# Step 1: Create 2 columns
tmux split-window -h

# Step 2: Split each column into 4 rows
tmux select-pane -t 0
tmux split-window -v
tmux split-window -v
tmux split-window -v

tmux select-pane -t 4
tmux split-window -v
tmux split-window -v
tmux split-window -v

# Set equal pane sizes
tmux select-layout tiled

# GPU 0: R3Det
tmux send-keys -t 0 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=0 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/r3det-oc_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/r3det-oc_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 1: GWD
tmux send-keys -t 1 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=1 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/gwd_rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/gwd_rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 2: ROI Transformer
tmux send-keys -t 2 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=2 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/roi-trans-le90_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/roi-trans-le90_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 3: Rotated ATSS
tmux send-keys -t 3 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=3 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-atss-le90_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/rotated-atss-le90_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 4: Rotated Faster R-CNN
tmux send-keys -t 4 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=4 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 5: Rotated FCOS
tmux send-keys -t 5 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=5 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-fcos-le90_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/rotated-fcos-le90_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 6: Rotated RetinaNet
tmux send-keys -t 6 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=6 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter

# GPU 7: S2ANet
tmux send-keys -t 7 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=7 python playground/comparison_detector/eval_mmrotate_detector_mapnc.py --pickle_result_path playground/comparison_detector/work_dirs/s2anet-le135_r50_fpn_1x_dota-512/output_val.pkl >> playground/comparison_detector/work_dirs/s2anet-le135_r50_fpn_1x_dota-512/eval_mmrotate_detector_mapnc.log " Enter