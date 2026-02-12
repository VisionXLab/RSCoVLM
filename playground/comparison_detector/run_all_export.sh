#!/bin/bash
# Method        score thr.
# GWD	        0.40
# R3Det	        0.45
# Roi Trans.	0.80
# ATSS	        0.35
# Faster RCNN	0.85
# FCOS	        0.30
# RetinaNet	    0.40
# S2ANet	    0.50

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

# # GPU 0: R3Det
# tmux send-keys -t 0 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=0 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/r3det-oc_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/r3det-oc_r50_fpn_1x_dota-512/dota_Task1_threshold0_45 --threshold 0.45 " Enter

# # GPU 1: GWD
# tmux send-keys -t 1 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=1 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/gwd_rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/gwd_rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota-512/dota_Task1_threshold0_40 --threshold 0.40 " Enter

# # GPU 2: ROI Transformer
# tmux send-keys -t 2 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=2 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/roi-trans-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/roi-trans-le90_r50_fpn_1x_dota-512/dota_Task1_threshold0_80 --threshold 0.80 " Enter

# # GPU 3: Rotated ATSS
# tmux send-keys -t 3 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=3 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-atss-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/rotated-atss-le90_r50_fpn_1x_dota-512/dota_Task1_threshold0_35 --threshold 0.35 " Enter

# # GPU 4: Rotated Faster R-CNN
# tmux send-keys -t 4 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=4 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota-512/dota_Task1_threshold0_85 --threshold 0.85 " Enter

# # GPU 5: Rotated FCOS
# tmux send-keys -t 5 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=5 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-fcos-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/rotated-fcos-le90_r50_fpn_1x_dota-512/dota_Task1_threshold0_30 --threshold 0.30 " Enter

# # GPU 6: Rotated RetinaNet
# tmux send-keys -t 6 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=6 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-512/dota_Task1_threshold0_40 --threshold 0.40 " Enter

# # GPU 7: S2ANet
# tmux send-keys -t 7 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=7 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/s2anet-le135_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/work_dirs/s2anet-le135_r50_fpn_1x_dota-512/dota_Task1_threshold0_50 --threshold 0.50 " Enter


# GPU 0: R3Det
tmux send-keys -t 0 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=0 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/r3det-oc_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/r3det-oc_r50_fpn_1x_dota-512_dota_Task1_threshold0_45 --threshold 0.45 " Enter

# GPU 1: GWD
tmux send-keys -t 1 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=1 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/gwd_rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/gwd_rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota-512_dota_Task1_threshold0_40 --threshold 0.40 " Enter

# GPU 2: ROI Transformer
tmux send-keys -t 2 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=2 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/roi-trans-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/roi-trans-le90_r50_fpn_1x_dota-512_dota_Task1_threshold0_80 --threshold 0.80 " Enter

# GPU 3: Rotated ATSS
tmux send-keys -t 3 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=3 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-atss-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/rotated-atss-le90_r50_fpn_1x_dota-512_dota_Task1_threshold0_35 --threshold 0.35 " Enter

# GPU 4: Rotated Faster R-CNN
tmux send-keys -t 4 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=4 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-faster-rcnn-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/rotated-faster-rcnn-le90_r50_fpn_1x_dota-512_dota_Task1_threshold0_85 --threshold 0.85 " Enter

# GPU 5: Rotated FCOS
tmux send-keys -t 5 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=5 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-fcos-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/rotated-fcos-le90_r50_fpn_1x_dota-512_dota_Task1_threshold0_30 --threshold 0.30 " Enter

# GPU 6: Rotated RetinaNet
tmux send-keys -t 6 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=6 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/rotated-retinanet-rbox-le90_r50_fpn_1x_dota-512_dota_Task1_threshold0_40 --threshold 0.40 " Enter

# GPU 7: S2ANet
tmux send-keys -t 7 "conda activate rscoagent && CUDA_VISIBLE_DEVICES=7 python playground/comparison_detector/export_new_submission.py --pickle_result_path playground/comparison_detector/work_dirs/s2anet-le135_r50_fpn_1x_dota-512/output_test.pkl --outfile_prefix playground/comparison_detector/submission/s2anet-le135_r50_fpn_1x_dota-512_dota_Task1_threshold0_50 --threshold 0.50 " Enter
