from .metric import (
    compute_qwen25vl_vlm_r1_rec_acc, compute_contain_acc, compute_cider, geochat_scores
)
from .vhm_bench import compute_vhm_scores
from .teochatlas_bench import compute_teochatlas_scores
from .dense_det import eval_dense_det
from .geoground import compute_geoground_acc
from .large_image_bench import mme_realworld_remote_sensing_scores, lrsvqa_scores


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


def evaluate_benchmark(data_pipe_list, benchmark, save_path, args):
    clean_benchmark, icl_shot, _ = check_benchmark(benchmark)
    if clean_benchmark in ['refcoco_val', 'refcocop_val', 'refcocog_val', 'refgta_subsample','dior_rsvg_val', 'vrsbench_referring']:
        return compute_qwen25vl_vlm_r1_rec_acc(data_pipe_list, prompt_type=args.grounding_prompt_type)
    # elif benchmark in ["vrsbench_caption", ]:
    #     return computer_caption_belu(data_pipe_list)
    elif benchmark.startswith("cls_"):
        return compute_contain_acc(data_pipe_list)
    elif clean_benchmark in ["vrsbench_vqa"]:
        return compute_contain_acc(data_pipe_list)
    elif clean_benchmark.startswith("mme_realworld_remote_sensing"):
        return mme_realworld_remote_sensing_scores(data_pipe_list)
    elif clean_benchmark.startswith("lrsvqa"):
        return lrsvqa_scores(data_pipe_list)
    elif clean_benchmark in ['vrsbench_caption', 'cap_nwpu_caption', 'cap_rsicd', 'cap_rsitmd', 'cap_sydney_caption', 'cap_ucm_caption']:
        return compute_cider(data_pipe_list)
    elif clean_benchmark in ['geochat_aid', 'geochat_grounding_description', 'geochat_hrben', 'geochat_lrben', 'geochat_referring', 'geochat_region_captioning', 'geochat_ucmerced']:
        return geochat_scores(data_pipe_list, clean_benchmark, prompt_type=args.grounding_prompt_type)  # TODO: unify the args -> prompt type with dense detection
    elif clean_benchmark.startswith('vhm'):
        return compute_vhm_scores(data_pipe_list)
    elif clean_benchmark.startswith('teochatlas'):
        return compute_teochatlas_scores(data_pipe_list)
    elif clean_benchmark.startswith('geoground'):
        return compute_geoground_acc(data_pipe_list, prompt_type=args.grounding_prompt_type)
    elif clean_benchmark.startswith('dota'):
        return eval_dense_det(
            data_pipe_list, clean_benchmark, save_path, 
            prompt_type=args.dense_det_prompt_type,
            eval_box_type=args.dense_det_eval_box_type, 
            clear_pred_for_empty_gt=args.dense_det_clear_pred_for_empty_gt
        )
    else:
        raise NotImplementedError(f"Unknown benchmark: {benchmark}")
