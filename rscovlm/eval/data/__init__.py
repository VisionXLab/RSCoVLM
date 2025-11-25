import os
import json
import random
import logging
from PIL import Image
from itertools import islice
from types import GeneratorType
from transformers.utils import cached_file

from .geochat_bench import prepare_geochat
from .scene_cls import prepare_scene_cls, prepare_scene_cls_local, prepare_scene_cls_timm_incontext
from .vrsbench import prepare_vrsbench
from .vhm_bench import prepare_vhm_eval
from .teochatlas_bench import prepare_teochatlas_eval
from .geoground import prepare_geoground_eval
from .dense_det import prepare_dense_det
from .large_image_bench import prepare_lrsvqa, prepare_mme_realworld_remote_sensing
from .data import prepare_vlm_r1_rec, prepare_rs_caption, prepare_rrsisd_res

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 8e8


def check_benchmark(benchmark):
    parts = benchmark.rsplit('_')
    unique = False # for cls task
    if parts and parts[-1].isdigit():
        shots = int(parts[-1])
        if 'unique' == parts[-2]:
            unique = True
            benchmark_name = '_'.join(parts[:-2])
        else:
            benchmark_name = '_'.join(parts[:-1])
        return benchmark_name, shots, unique
    return benchmark, 0, unique


def get_min_and_max_pixels(args):
    min_pixels, max_pixels = None, None
    args_min_pixels, args_max_pixels = getattr(args, "min_pixels", None), getattr(args, "max_pixels", None)

    if args_min_pixels is None or args_max_pixels is None:
        # preprocess_config_path = cached_file(args.model_ckpt_path, "preprocessor_config.json")
        assert len(args.model_ckpt_path) == 1, "Only support one model ckpt path for now"
        preprocess_config_path = os.path.join(args.model_ckpt_path[0], "preprocessor_config.json")
        if preprocess_config_path is not None:
            with open(preprocess_config_path, "r") as f:
                preprocess_config = json.load(f)
            min_pixels = preprocess_config.get("min_pixels", None)
            max_pixels = preprocess_config.get("max_pixels", None)
    
    if args_min_pixels is not None:
        min_pixels = args_min_pixels
    if args_max_pixels is not None:
        max_pixels = args_max_pixels
    return min_pixels, max_pixels


def prepare_data_pipe(benchmark, args):
    logger.info(f"Preparing data pipe for {benchmark}...")
    benchmark = str(benchmark).strip().lower()
    clean_benchmark, icl_shot, unique = check_benchmark(benchmark)  # `icl` indicates in-context learning

    if clean_benchmark in ['refcoco_val', 'refcocop_val', 'refcocog_val', 'refgta_subsample', 'dior_rsvg_val', 'rsvg_val']:
        data_pipe_list = prepare_vlm_r1_rec(clean_benchmark, args.vlm_r1_rec_json_root, args.vlm_r1_rec_image_root, args.grounding_prompt_type)

    elif clean_benchmark in ['cls_resisc', 'cls_aid']:
        image_root = {
            'cls_resisc': args.nwpu_resisc45_root,
            'cls_aid': args.aid_root
        }[clean_benchmark]
        data_pipe_list = prepare_scene_cls_local(image_root, clean_benchmark, icl_shot, args.scene_cls_json_root_local, args.ppl_json_path, unique)

    elif clean_benchmark in ['cls_aid_full']:
        data_pipe_list = prepare_scene_cls(args.aid_root, from_timm=False)

    elif clean_benchmark in ['cls_resisc_timm', 'cls_eurosat_timm']:
        data_root = {
            'cls_resisc_timm': 'timm/resisc45',
            'cls_eurosat_timm': 'timm/eurosat-rgb'
        }[clean_benchmark]
        if icl_shot == 0:
            data_pipe_list = prepare_scene_cls(data_root, from_timm=True)
        else:
            data_pipe_list = prepare_scene_cls_timm_incontext(data_root, num_incontext_samples=icl_shot, unique_class_mode=unique)
    
    elif clean_benchmark in ['vrsbench_caption', 'vrsbench_referring', 'vrsbench_vqa']:
        data_pipe_list = prepare_vrsbench(args.vrsbench_image_root, args.vrsbench_data_root, clean_benchmark, args.grounding_prompt_type, icl_shot, args.ppl_json_path)

    elif clean_benchmark in ['cap_nwpu_caption', 'cap_rsicd', 'cap_rsitmd', 'cap_sydney_caption', 'cap_ucm_caption']:
        benchmark_to_image_root = {
            'cap_nwpu_caption': args.nwpu_caption_image_root,
            'cap_rsicd': args.rsicd_image_root,
            'cap_rsitmd': args.rsitmd_image_root,
            'cap_sydney_caption': args.sydney_caption_image_root,
            'cap_ucm_caption': args.ucm_caption_image_root
        }
        image_root = benchmark_to_image_root[clean_benchmark]
        data_pipe_list = prepare_rs_caption(image_root, args.rs_caption_json_root, clean_benchmark, icl_shot, args.ppl_json_path)
    
    elif clean_benchmark in ['res_rrsisd_refer', 'res_rrsisd_xml']:
        data_pipe_list = prepare_rrsisd_res(args.rrsisd_root, ann_type=clean_benchmark.split('_')[-1])

    elif clean_benchmark in ['geochat_aid', 'geochat_grounding_description', 'geochat_hrben', 'geochat_lrben', 'geochat_referring', 'geochat_region_captioning', 'geochat_ucmerced']:
        data_pipe_list = prepare_geochat(args.geochat_image_root, args.geochat_data_root, clean_benchmark, icl_shot, args.ppl_json_path, args.grounding_prompt_type, args.aid_root, args.geochat_ucmerced_image_root, args.hrben_image_root, unique)

    elif clean_benchmark.startswith('mme_realworld_remote_sensing'):
        data_pipe_list = prepare_mme_realworld_remote_sensing(
            args.mme_realworld_root, visual_cot=clean_benchmark.endswith('visual_cot'))

    elif clean_benchmark.startswith('lrsvqa'):
        data_pipe_list = prepare_lrsvqa(
            args.lrsvqa_jsonl_path, args.lrsvqa_image_root, 
            visual_cot=clean_benchmark.endswith('visual_cot'))

    elif clean_benchmark.lower().startswith("vhm"):
        data_pipe_list = prepare_vhm_eval(args.vhm_root, benchmark)

    elif clean_benchmark.lower().startswith("teochatlas"):
        min_pixels, max_pixels = get_min_and_max_pixels(args)
        data_pipe_list = prepare_teochatlas_eval(
            args.teochatlas_root, benchmark, args.grounding_prompt_type, 
            min_pixels=min_pixels, max_pixels=max_pixels, 
        )

    elif clean_benchmark.lower().startswith("geoground"):
        data_pipe_list = prepare_geoground_eval(args.geoground_root, benchmark, args.grounding_prompt_type)

    elif clean_benchmark.lower().startswith("dota"):
        data_pipe_list = prepare_dense_det(
            clean_benchmark, 
            pred_box_type=args.dense_det_pred_box_type, 
            eval_box_type=args.dense_det_eval_box_type, 
            prompt_type=args.dense_det_prompt_type
        )

    else:
        raise ValueError(f"Unknown benchmark {benchmark}")

    if args.shuffle_seed is not None:
        if isinstance(data_pipe_list, GeneratorType):
            data_pipe_list = list(data_pipe_list)
        random.Random(args.shuffle_seed).shuffle(data_pipe_list)
    if args.sample_num is not None:
        if isinstance(data_pipe_list, GeneratorType):
            data_pipe_list = islice(data_pipe_list, args.sample_num)
        else:
            data_pipe_list = data_pipe_list[:args.sample_num]
    logger.info(f"Data pipe prepared for {benchmark}" + (f" with {len(data_pipe_list)} samples" if isinstance(data_pipe_list, list) else ""))
    return data_pipe_list
