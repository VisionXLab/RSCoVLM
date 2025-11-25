"""
import json
resisc_get_cat = lambda x: x.split("_")[0]
with open("./ppl_resisc45_train.json", "r") as f:
    data = json.load(f)
data = [(k, v, resisc_get_cat(k)) for k, v in data.items()]
data = sorted(data, key=lambda x: x[1], reverse=True)
print([s[0] for s in data[:32]])
samples_unique_cat = []
cat_set = set()
for s in data:
    if s[2] not in cat_set:
        cat_set.add(s[2])
        samples_unique_cat.append(s)
print([s[0] for s in samples_unique_cat[:32]])
"""
import os
import json
from tqdm import tqdm
import re

import torch
import torch.distributed as dist

import datasets
from rscovlm.eval.inferencer import get_inferencer
from rscovlm.utils import init_distributed_device, world_info_from_env #, partition_for_rank
from rscovlm.utils.qwen_vl_utils import process_vision_info

def calculate_single_round_conversation_ppl(benchmark, inferencer, x, image_root, prompt_type='json'): # for a single data item
    # grounding
    image_path = os.path.join(image_root, x["image_id"])
    ppl_results = [] # single img single solution
    if prompt_type == 'json':
        msg_instruction = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": f"Locate the {x['normal_caption']}, output the bbox coordinates using JSON format"}
            ]}
        ]
    elif prompt_type == 'plain':
        msg_instruction = [
            # {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": f"find the {x['normal_caption']}, delivering coordinates in plain text format 'x1,y1,x2,y2 object'"}
            ]}
        ]
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    # image_sizes: (original_width, original_height, resized_width, resized_height)
    pil_imgs, video_inputs, image_sizes = process_vision_info(msg_instruction, return_image_sizes=True) # need a list input

    original_width, original_height, resized_width, resized_height = image_sizes[0]
    scale_w = resized_width / original_width  # 
    scale_h = resized_height / original_height  # 

    bboxes = x["solution"]
    if len(bboxes) < 4: # vrsbench has bbox w/. len(bboxes) = 3
        return None, None, None
    converted_int_bboxes = [
            int(bboxes[0] * scale_w), # x1
            int(bboxes[1] * scale_h), # y1
            int(bboxes[2] * scale_w), # x2
            int(bboxes[3] * scale_h) # y2
            ]
    if prompt_type == 'json': # json
        answer = f"""```json\n[\n{{"bbox_2d": {converted_int_bboxes}, "label": "{x['normal_caption']}"}}\n]\n```"""
        gt_list = [answer]
        ## check
        # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
        # bbox_match = re.search(bbox_pattern, answer)
    elif prompt_type == 'plain':
        answer = f"{converted_int_bboxes[0]},{converted_int_bboxes[1]},{converted_int_bboxes[2]},{converted_int_bboxes[3]} {x['normal_caption']}\n"
        gt_list = [answer]
        ## check
        # pattern = r"(\d+),(\d+),(\d+),(\d+)\s+(.+?)\s*$"
        # bbox_match = re.match(pattern, answer.strip().split('\n')[0].strip())
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    # =======================================================================================================================
    text_instruction = inferencer.processor.apply_chat_template(msg_instruction, tokenize=False, add_generation_prompt=True)

    instruction_inputs = inferencer.processor(
        text=[text_instruction],
        images=pil_imgs, # a list
        return_tensors="pt",
        padding=True
    )
    assert torch.all(instruction_inputs.attention_mask)
    instruction_inputs["labels"] = torch.full_like(instruction_inputs["input_ids"], -100)
    all_msg_response = []
    for answer in gt_list: # 1 image multi captions
        msg_response = [{"role": "assistant", "content": answer}]
        text_full = inferencer.processor.apply_chat_template(
            msg_instruction + msg_response,
            tokenize=False
        ).rstrip("\n")
        text_response = text_full.replace(text_instruction, "")
        # pil_imgs = [pil_image]
        # tokenize the answer
        response_inputs = inferencer.processor.tokenizer(
            text=[text_response],
            return_tensors="pt",
            padding="longest",
            padding_side="right",
            truncation=True
        )
        response_labels = response_inputs["input_ids"].clone()
        response_labels[~response_inputs["attention_mask"].bool()] = -100
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        inputs = {
            "input_ids": torch.cat([instruction_inputs["input_ids"].to(device),response_inputs["input_ids"].to(device)], dim=1),
            "attention_mask": torch.cat([instruction_inputs["attention_mask"].to(device),response_inputs["attention_mask"].to(device)], dim=1),
            "labels": torch.cat([instruction_inputs["labels"].to(device),response_labels.to(device)], dim=1),
            }
        for key in instruction_inputs.keys():
            if key not in inputs:
                inputs[key] = instruction_inputs[key].to(device) if torch.is_tensor(instruction_inputs[key]) else instruction_inputs[key]
        with torch.no_grad():
            loss = inferencer.model(**inputs).loss
            ppl = torch.exp(loss).item()
            ppl_results.append(ppl)
            all_msg_response.append(msg_response)
            del inputs, response_inputs, response_labels
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    # =======================================================================================================================
    return ppl_results, msg_instruction, all_msg_response # captions fot a single image
    


def gather_list(list_to_gather, rank, world_size):
    rank_placeholder = [None for _ in range(world_size)]
    dist.barrier()
    print(f"Rank {rank} is gathering list data")
    dist.all_gather_object(rank_placeholder, list_to_gather)
    if rank == 0:
        output_list = []
        for gathered_list in rank_placeholder:
            output_list.extend(gathered_list)
        return output_list
    else:
        return None
    
def partition_for_rank(all_rank_item_list: list, rank: int, world_num: int) -> list:
    if isinstance(all_rank_item_list, list):
        this_rank_item_list = []
        this_rank_index = range(rank, len(all_rank_item_list), world_num)
        for idx in this_rank_index:
            this_rank_item_list.append(all_rank_item_list[idx])
        return this_rank_item_list

    else:
        raise ValueError("The input type is not supported.")


def incontext_max_ppl(model_path, benchmark, image_root, json_path, save_path):
    _, rank, world_size, _ = world_info_from_env()
    device = init_distributed_device(dist_backend=None) # using nccl may raise error for 4090 cluster

    inferencer = get_inferencer(
        model_path, 
        device=device, 
        torch_dtype="bf16", 
        attn_implementation="flash_attention_2", 
        lazy_loading=False,
    )
    if benchmark != "vrsbench_referring":
        json_name = f"{'_'.join(benchmark.rsplit('_')[:-1])}_train.json"
    else:
        json_name = f"{benchmark}_train.json"
    train_data = json.load(open(os.path.join(json_path, json_name), "r"))
    
    train_data = partition_for_rank(train_data, rank, world_size)

    ppl_root = os.path.join(save_path, 'ppl')
    os.makedirs(ppl_root, exist_ok=True)
    ppl_file = os.path.join(ppl_root, f"ppl_{benchmark}_train.json")
    # save flatted [{image_id: ..., "gt_idx": ..., "ppl": ...}]
    all_ppl_entries = []
    for x in tqdm(train_data, disable=(rank!=0), desc="PPL..."):
        if benchmark in ['refcoco_val', 'refcocop_val', 'refcocog_val', 'refgta_subsample', 'dior_rsvg_val']:
            x['image_id'] = x['image']
        gt_ppls, msg_instruction, all_msg_response = calculate_single_round_conversation_ppl(benchmark, 
            inferencer, x, image_root, prompt_type)
        if not gt_ppls:
            continue
        for gt_idx, (ppl, ans) in enumerate(zip(gt_ppls, all_msg_response)):
            all_ppl_entries.append({
                "image_id": x["image_id"],
                "gt_idx": gt_idx,
                "ppl": ppl,
                "msg_response": ans, # list
                "msg_instruction": [msg_instruction[-1]] # only user, convert to list
            })
    
    if world_size > 1:
        all_ppl_entries = gather_list(all_ppl_entries, rank, world_size)

    if rank == 0:
        all_ppl_entries.sort(key=lambda x: x["ppl"], reverse=True)
        with open(f"{ppl_file}", "w") as f:
            json.dump(all_ppl_entries, f, indent=4, ensure_ascii=False)
        print(f"save sorted ppl to {ppl_file}")

    # top_k_pairs = [entry for entry in all_ppl_entries[:in_context_shot]]
    # return top_k_pairs


if __name__ == '__main__':
    model_path = "./output/qwen2-5-vl-ins_refgeo-vhm_dora-eva-r64_pix256-1296_lr0.0002/checkpoint-5863"
    json_path = "/mnt/petrelfs/liqingyun/VLM-R1/rec_jsons_processed"
    benchmarks = ['refcoco_val', 'refcocop_val', 'refcocog_val', 'refgta_subsample', 'dior_rsvg_val', "vrsbench_referring"]
    all_image_root = "/mnt/petrelfs/liqingyun/VLM-R1/images"
    # "./playground/data/VRSBench/Images"
    # 

    save_path = "./playground/incontext_examples"
    # prepare_for_scene_cls_timm(model_path, "timm/resisc45")
    benchmark = 'refcoco_val'
    image_root = all_image_root
    prompt_type = 'json'
    incontext_max_ppl(model_path, benchmark, image_root, json_path, save_path)
