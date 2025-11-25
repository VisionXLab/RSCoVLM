# import sys
# sys.path.insert(0, "/mnt/petrelfs/liqingyun/msr/code/rscovlm")
import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
from collections.abc import Iterable

from PIL import Image

import torch.distributed as dist

from rscovlm.utils import (
    init_distributed_device, world_info_from_env, partition_for_rank, 
    prepare_logger_for_rank, gather_list, maybe_use_hf_mirror
)
from .inferencer import get_inferencer
from .data import prepare_data_pipe
from .metric import evaluate_benchmark
from .params import parse_args, max_batch_sizes

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_rscovlm/"

logger = logging.getLogger(__name__)
    

def check_whether_inference_done(save_path, benchmarks):
    for benchmark in benchmarks:
        if not os.path.exists(os.path.join(save_path, benchmark, "inference_results.json")):
            return False
    return True


def clear_image_from_data_pipe(data_pipe_list):
    def _convert(obj):
        if isinstance(obj, Image.Image):
            return repr(obj)

    for sample in data_pipe_list:
        for m in sample['message']:
            for content in m['content']:
                if isinstance(content, str):
                    continue
                if content['type'] == 'image' and not isinstance(content['image'], str):
                    content['image'] = _convert(content['image'])

        if 'pil_image' in sample:
            sample['pil_image'] = _convert(sample['pil_image'])
        if 'image' in sample:
            sample['image'] = _convert(sample['image'])
    return data_pipe_list


def inference(model_ckpt_path, save_path, args, device):
    if check_whether_inference_done(save_path, args.benchmarks):
        return
    
    model = get_inferencer(
        model_ckpt_path, 
        device=device, 
        torch_dtype=args.torch_dtype, 
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_vllm=args.use_vllm,
        max_max_length=args.max_max_length,
        mimo_no_think=args.mimo_no_think,
    )
    for benchmark in args.benchmarks:
        if check_whether_inference_done(save_path, [benchmark]):
            continue
        
        data_pipe_list = prepare_data_pipe(benchmark, args)
        
        if args.world_size > 1:
            data_pipe_list = partition_for_rank(data_pipe_list, args.rank, args.world_size)

        # inference
        batch_size = min(max_batch_sizes.get(benchmark, args.batch_size), args.batch_size)
        if isinstance(data_pipe_list, list):
            for i in tqdm(range(0, len(data_pipe_list), batch_size), desc=f"Inferencing (list) (rank {args.rank} | {args.world_size}) {benchmark}"):
                model(data_pipe_list[i:i + batch_size])
        elif isinstance(data_pipe_list, Iterable):
            all_data_pipe_list = []
            batch_data_pipe_list = []
            for data_pipe in tqdm(data_pipe_list, disable=(args.rank!=0), desc=f"Inferencing (Iterable) {benchmark}"):  # no total
                if len(batch_data_pipe_list) < batch_size:
                    batch_data_pipe_list.append(data_pipe)
                    all_data_pipe_list.append(data_pipe)
                if len(batch_data_pipe_list) >= batch_size:
                    model(batch_data_pipe_list)
                    batch_data_pipe_list = []
            if len(batch_data_pipe_list) > 0:
                model(batch_data_pipe_list)
            data_pipe_list = all_data_pipe_list
        else:
            raise ValueError("data_pipe_list should be either list or generator")
        clear_image_from_data_pipe(data_pipe_list)
        assert len(data_pipe_list) > 0, f"Empty data_pipe_list for {benchmark}"

        if args.world_size > 1:
            data_pipe_list = gather_list(data_pipe_list, args.rank, args.world_size)

        if args.rank == 0:
            os.makedirs(os.path.join(save_path, benchmark), exist_ok=True)
            with open(os.path.join(save_path, benchmark, "inference_results.json"), "w") as f:
                json.dump(data_pipe_list, f, indent=4, ensure_ascii=False)


def evaluate(save_path, args):
    results = {}
    for benchmark in args.benchmarks:
        data_pipe_list = json.load(open(os.path.join(save_path, benchmark, "inference_results.json"), "r"))
        results[benchmark] = evaluate_benchmark(data_pipe_list, benchmark, save_path, args)
    return results


def save_and_print_results(results, save_path, args):
    json_results = json.dumps(results, indent=4, ensure_ascii=False)
    logger.info(json_results)
    with open(os.path.join(save_path, "res_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"), "w") as f:
        f.write(json_results)
        f.write("\n")
        f.write(json.dumps(args.__dict__, indent=4, ensure_ascii=False))


def main(args):
    maybe_use_hf_mirror()
    args.local_rank, args.rank, args.world_size, args.distributed_type = world_info_from_env()
    device = init_distributed_device(dist_backend=None, timeout_min=1440)  # using nccl may raise error for 4090 cluster
    prepare_logger_for_rank(logger, args.rank)

    try:
        for idx, (model_ckpt_path, save_path) in enumerate(zip(args.model_ckpt_path, args.save_path)):
            logger.info(f"[{idx} / {len(args.model_ckpt_path)}] Inference for {model_ckpt_path}")
            inference(model_ckpt_path, save_path, args, device)

        if not args.pass_evaluate and args.rank == 0:
            for idx, save_path in enumerate(args.save_path):
                logger.info(f"[{idx} / {len(args.save_path)}] Evaluate for {save_path}")
                results = evaluate(save_path, args)
                save_and_print_results(results, save_path, args)
        
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
