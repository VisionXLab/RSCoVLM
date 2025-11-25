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
from rscovlm.utils import init_distributed_device, world_info_from_env, partition_for_rank
from rscovlm.utils.qwen_vl_utils import process_vision_info

def calculate_single_round_conversation_ppl(benchmark, inferencer, x, image_root, prompt_type='json', rec_normalized=False): # for a single data item
    if benchmark in ['vrsbench_caption', 'cap_nwpu_caption', 'cap_rsicd', 'cap_rsitmd', 'cap_sydney_caption', 'cap_ucm_caption']: 
        image_path = os.path.join(image_root, x["image_id"])
        question = x["question"]
        ppl_results = [] # save ppl for all captions
        gt_list = x['ground_truth']
        msg_instruction = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]},
        ]
        pil_imgs, video_inputs, image_sizes = process_vision_info(msg_instruction, return_image_sizes=True) # need a list input
    elif benchmark in ['refcoco_val', 'refcocop_val', 'refcocog_val', 'refgta_subsample', 'dior_rsvg_val']:
        # grounding
        image_path = os.path.join(image_root, x["image"])
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

        if rec_normalized:
            bboxes = x["normalized_solution"] # TODO: enable normalized solution
        else:
            bboxes = x["solution"]
            converted_int_bboxes = [
                    int(bboxes[0] * scale_w), # x1
                    int(bboxes[1] * scale_h), # y1
                    int(bboxes[2] * scale_w), # x2
                    int(bboxes[3] * scale_h) # y2
                    ]
            if prompt_type == 'json': # json
                answer = answer = f"""```json\n[\n{{"bbox_2d": {converted_int_bboxes}, "label": "{x['normal_caption']}"}}\n]\n```"""
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
    


def gather_dict(dict_to_gather, rank, world_size):
    rank_placeholder = [None for _ in range(world_size)]
    dist.barrier()
    print(f"Rank {rank} is gathering")
    dist.all_gather_object(rank_placeholder, dict_to_gather)

    if rank == 0:
        output_dict = {}
        for gathered_dict in rank_placeholder:
            for k, v in gathered_dict.items():
                output_dict[k] = v
        return output_dict
    else:
        return None
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
    


def incontext_max_ppl(train_data, image_root, inferencer, benchmark, in_context_shot, save_path="", prompt_type='plain'):
    ppl_root = os.path.join(save_path, 'ppl')
    os.makedirs(ppl_root, exist_ok=True)
    ppl_file = os.path.join(ppl_root, f"ppl_{benchmark.split('/')[-1]}_train.json")
    if not os.path.exists(ppl_file):
        _, rank, world_size, _ = world_info_from_env()
        train_data = partition_for_rank(train_data, rank, world_size)
        # save flatted [{image_id: ..., "gt_idx": ..., "ppl": ...}]
        all_ppl_entries = []
        for x in tqdm(train_data, disable=(rank!=0), desc="PPL..."):
            if benchmark in ['refcoco_val', 'refcocop_val', 'refcocog_val', 'refgta_subsample', 'dior_rsvg_val']:
                x['image_id'] = x['image']
            gt_ppls, msg_instruction, all_msg_response = calculate_single_round_conversation_ppl(benchmark, 
                inferencer, x, image_root, prompt_type)
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
        all_ppl_entries.sort(key=lambda x: x["ppl"], reverse=True)

        if rank == 0:
            with open(f"{ppl_file}", "w") as f:
                json.dump(all_ppl_entries, f, indent=4, ensure_ascii=False)
            print(f"save sorted ppl to ppl_{benchmark.split('/')[-1]}_train.json")
    else:
        with open(ppl_file, "r") as f:
            all_ppl_entries = json.load(f)
    top_k_pairs = [entry for entry in all_ppl_entries[:in_context_shot]]
    return top_k_pairs


# if __name__ == '__main__':
#     model_path = "/mnt/hwfile/share_data/liqingyun/Qwen2.5-VL-3B-Instruct"
#     # prepare_for_scene_cls_timm(model_path, "timm/resisc45")
#     prepare_for_scene_cls_timm(model_path, "timm/eurosat-rgb")
