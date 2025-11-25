import os
import re
import logging
import argparse

logger = logging.getLogger(__name__)
    

default_args = dict(
    model_ckpt_path="./playground/Qwen2.5-VL-3B-Instruct", 
    vlm_r1_rec_json_root="./playground/data/rec_jsons_processed",  # DIOR: 800x800
    vlm_r1_rec_image_root="./playground/data/detection",
    scene_cls_json_root_local="./playground/data/scene_cls",
    nwpu_resisc45_root="./playground/data/scene_cls/NWPU-RESISC45",  # 256x256
    aid_root="./playground/data/scene_cls/AID",  # 600x600
    vrsbench_image_root="./playground/data/VRSBench/Images",
    vrsbench_data_root="./playground/data/VRSBench",
    rs_caption_json_root="./playground/data/rs_caption/rs_caption_jsons/converted",
    nwpu_caption_image_root="./playground/data/scene_cls/NWPU-RESISC45",
    rsicd_image_root="./playground/data/rs_caption/rsicd/RSICD_images",
    rsitmd_image_root="./playground/data/rs_caption/rsitmd/images",
    sydney_caption_image_root="./playground/data/rs_caption/sydney_caption/imgs",
    ucm_caption_image_root="./playground/data/rs_caption/ucm_caption/imgs",
    ppl_json_path="./playground/incontext_examples/ppl",
    rrsisd_root="./playground/data/RRSIS-D",
    geochat_image_root="./playground/data/refGeo/images/geochat",
    geochat_data_root="./playground/data/geochat_data/benchmark/converted/",
    geochat_ucmerced_image_root="./playground/data/scene_cls/UCMerced_LandUse/Images",
    hrben_image_root="./playground/data/rsvqa/RSVQA_HR/Data",
    lrsvqa_jsonl_path="./playground/data/LRS_VQA/annotations/benchmark/LRS_VQA_merged.jsonl",
    lrsvqa_image_root="./playground/data/LRS_VQA/images/benchmark",
    mme_realworld_root="./playground/data/MME-RealWorld/",
    vhm_root="./playground/data/VHM_eval_dataset/",
    teochatlas_root="./playground/data/TEOChatlas/",
    geoground_root="./playground/data/refGeo/",
)


max_batch_sizes = {
    'cap_nwpu_caption_16': 16,
}


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    # model options
    parser.add_argument("--model_ckpt_path", type=str, default=None, nargs="+", 
                        help="Path to model ckpt, you can set multiple ones. If is not set, use default Qwen2.5VL-3B-Instruct path.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", 
                        help="torch_dtype for transformers model. If set, modify for all the ckpt to be test.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", 
                        help="attn_implementation for transformers model. If set, modify for all the ckpt to be test.")
    parser.add_argument("--min_pixels", type=int, default=None, 
                        help="Minimum pixels for image. If set, modify for all the ckpt to be test.")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Maximum pixels for image. If set, modify for all the ckpt to be test.")
    parser.add_argument("--use_vllm", action="store_true", default=False, 
                        help="If set, use vllm for inference.")
    parser.add_argument("--max_max_length", type=int, default=16384, 
                        help="Max max_length for inference.")
    parser.add_argument("--mimo_no_think", action="store_true", default=False, 
                        help="If set, add /no_think to the end of the user message.")
    # running options
    parser.add_argument("--save_path", type=str, default=None, nargs="+", 
                        help="Path to save inference results, you should set multiple ones if multiple model_ckpt_path are provided.")
    parser.add_argument("--eval_intermediate_checkpoints", action="store_true", default=False, 
                        help="If set, evaluate final checkpoint along with the multiple intermediate checkpoints.")
    parser.add_argument("--pass_evaluate", action="store_true", default=False, 
                        help="If set, only inference for results and pass the evaluation.")
    parser.add_argument("--folder_name", type=str, default="eval", help="folder name for saving evaluation results")
    # benchmark options
    parser.add_argument("--benchmarks", type=str, default=None, nargs="+", 
                        help="benchmark names, you can set multiple ones.")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for inference.")
    parser.add_argument("--shuffle_seed", type=int, default=None, 
                        help="If set, shuffle the dataset with this seed.")
    parser.add_argument("--sample_num", type=int, default=None, 
                        help="If set, clip the number of samples in the dataset.")
    parser.add_argument("--vlm_r1_rec_json_root", type=str, default=None)
    parser.add_argument("--vlm_r1_rec_image_root", type=str, default=None)
    parser.add_argument("--nwpu_resisc45_root", type=str, default=None)
    parser.add_argument("--aid_root", type=str, default=None)
    parser.add_argument("--vrsbench_image_root", type=str, default=None)
    parser.add_argument("--vrsbench_data_root", type=str, default=None)
    parser.add_argument("--rs_caption_json_root", type=str, default=None, help="Root directory for RS caption JSON files")
    parser.add_argument("--nwpu_caption_image_root", type=str, default=None)
    parser.add_argument("--rsicd_image_root", type=str, default=None)
    parser.add_argument("--rsitmd_image_root", type=str, default=None)
    parser.add_argument("--sydney_caption_image_root", type=str, default=None)
    parser.add_argument("--ucm_caption_image_root", type=str, default=None)
    parser.add_argument("--geoground_root", type=str, default=None)
    #
    parser.add_argument("--ppl_json_path", type=str, default=None, help="path to sorted ppl files")
    parser.add_argument("--scene_cls_json_root_local", type=str, default=None, help="path to scene cls annotation files")
    parser.add_argument("--rrsisd_root", type=str, default=None, help="path to rrsisd root")
    parser.add_argument("--geochat_image_root", type=str, default=None, help="path to geochat image root")
    parser.add_argument("--geochat_data_root", type=str, default=None, help="path to geochat json root")
    parser.add_argument("--geochat_ucmerced_image_root", type=str, default=None, help="path to unmerced images")
    parser.add_argument("--hrben_image_root", type=str, default=None, help="path to hrben images")
    parser.add_argument("--lrsvqa_jsonl_path", type=str, default=None, help="path to lrsvqa jsonl")
    parser.add_argument("--lrsvqa_image_root", type=str, default=None, help="path to lrsvqa image root")
    parser.add_argument("--mme_realworld_root", type=str, default=None, help="path to mme realworld root")
    parser.add_argument("--vhm_root", type=str, default=None, help="path to vhm root")
    parser.add_argument("--teochatlas_root", type=str, default=None, help="path to teochatlas root")
    #
    parser.add_argument("--grounding_prompt_type", type=str, default="plain", 
                        help="prompt type for grounding task, can be json, or plain")
    parser.add_argument("--dense_det_pred_box_type", type=str, default="qbox", 
                        help="pred box type for dense detection task, can be qbox, hbox, or rbox")
    parser.add_argument("--dense_det_eval_box_type", type=str, default="rbox", 
                        help="eval box type for dense detection task, can be qbox, hbox, or rbox")
    parser.add_argument("--dense_det_prompt_type", type=str, default="plain", 
                        help="prompt type for dense detection task, can be json, or plain")
    parser.add_argument("--dense_det_clear_pred_for_empty_gt", action="store_true", default=False,
                        help="If set, clear pred for empty gt in dense detection task.")
    
    args, leftovers = parser.parse_known_args()
    left_args = {leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)}
    args.left_args = left_args
    
    # set default args
    for key, value in default_args.items():
        if getattr(args, key) is None:
            if key == 'model_ckpt_path':
                args.model_ckpt_path = [value]
                args.save_path = ["./playground/eval_qwen25vl3b_baseline"]
            else:
                setattr(args, key, value)
    
    if args.eval_intermediate_checkpoints:
        assert args.save_path is None, "save_path is disable when evaluating intermediate checkpoints"
        new_model_ckpt_path = []
        new_save_path = []
        for path in args.model_ckpt_path:
            new_model_ckpt_path.append(path)
            new_save_path.append(os.path.join(path, args.folder_name))
            checkpoint_folder_list = [f for f in os.listdir(path) if re.match(r"checkpoint-\d+", f)]
            new_model_ckpt_path.extend([os.path.join(path, f) for f in checkpoint_folder_list])
            new_save_path.extend([os.path.join(path, f, args.folder_name) for f in checkpoint_folder_list])
        
        args.model_ckpt_path = new_model_ckpt_path
        args.save_path = new_save_path
        logger.info(f"Found {len(new_model_ckpt_path)} intermediate checkpoints")

    # specific result path can be assigned, or set it automatically
    if args.save_path is not None:
        assert len(args.save_path) == len(args.model_ckpt_path), \
            f"Length of save_path {args.save_path} should be the same as model_ckpt_path {args.model_ckpt_path}"
    else:
        args.save_path = [os.path.join(p, args.folder_name) for p in args.model_ckpt_path]

    if len(kwargs) > 0:
        for k, v in kwargs.items():
            setattr(args, k, v)
    return args
