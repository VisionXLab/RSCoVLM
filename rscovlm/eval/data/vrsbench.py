import os
import json
import logging

from .data import get_incontext_msg

logger = logging.getLogger(__name__)


def prepare_vrsbench(image_root, data_root, benchmark, prompt_type, icl_shot, ppl_json_path):
    task = benchmark.split('_')[-1]
    logger.info(f"Processing {task}...")
    filename = {
        "caption": "VRSBench_EVAL_Cap.json",
        "referring": f"{benchmark}_test.json",  #"VRSBench_EVAL_referring.json",  # TODO: support this
        "vqa": "VRSBench_EVAL_vqa.json",
    }[task]

    if task == "caption":
        incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)
    elif task == "referring":
        incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)
    elif task == 'vqa': 
        incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)
    data = json.load(open(os.path.join(data_root, filename), "r"))

    for idx, x in enumerate(data):
        image_path = os.path.join(image_root, x['image_id'])

        if task == "vqa":
            question = x['question']
            question += " Answer the question using a single word or a short phrase."
            message = incontext_msg + [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question}
                ]}
            ]
        elif task == 'caption':
            question = x['question']
            if isinstance(x.get("ground_truth"), str):
                x["ground_truth"] = [x["ground_truth"]] # need a list in evaluation
            message = incontext_msg + [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question}
                ]}
            ]
        elif task == "referring":
            if prompt_type == 'json':
                message = incontext_msg + [
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": f"Locate the {x['normal_caption']}, output the bbox coordinates using JSON format"}
                    ]}
                ]
            elif prompt_type == 'plain':
                message = incontext_msg + [
                    # {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": f"find the {x['normal_caption']}, delivering coordinates in plain text format 'x1,y1,x2,y2 object'"}
                    ]}
                ]
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

        x["message"] = message
        x["idx"] = idx
    return data
