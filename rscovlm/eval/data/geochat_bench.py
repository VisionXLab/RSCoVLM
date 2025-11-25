import os
import logging
from .data import get_incontext_msg, load_json_or_jsonl

logger = logging.getLogger(__name__)


def prepare_geochat(image_root, data_root, benchmark, icl_shot, ppl_json_path, prompt_type, aid_path=None, ucmerced_path=None, hrben_path=None, unique=False):
    # icl_chot, unique, ppk_json_path: for incontext-learning (caption, vqa)

    logger.info(f"Processing {benchmark}...")
    benchmark_info = {
        'geochat_aid':{'filename': 'aid.jsonl', 'task': 'cls'}, 
        'geochat_grounding_description': {'filename': 'grounding_description.jsonl', 'task': 'caption'},  #
        'geochat_hrben': {'filename': 'hrben.jsonl', 'task': 'vqa'}, #
        'geochat_lrben': {'filename': 'lrben.jsonl', 'task': 'vqa'}, #
        'geochat_referring': {'filename': 'referring.jsonl', 'task': 'referring'},  #
        'geochat_region_captioning': {'filename': 'region_captioning.jsonl', 'task': 'region_caption'}, 
        'geochat_ucmerced': {'filename': 'UCmerced.jsonl', 'task': 'cls'}
    }[benchmark]

    if benchmark_info["task"] == "vqa" or benchmark_info["task"] == "caption":
        # only vqa and caption need in-context
        incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)

    if "aid" in benchmark_info["filename"].lower():
        # aid_path: geochat doesnt contain aid images
        image_root = aid_path
    elif "ucmerced" in benchmark_info["filename"].lower():
        image_root = ucmerced_path
    elif "hrben" in benchmark_info['filename'].lower():
        image_root = hrben_path
    elif "lrben" in benchmark_info['filename'].lower():
        image_root = os.path.join(image_root, "LRBEN")

    data = load_json_or_jsonl(data_root, benchmark_info["filename"])

    for idx, x in enumerate(data):

        image_path = os.path.join(image_root, x['image_id'])

        if benchmark_info["task"] == "vqa":
            question = x['question']
            question += " Answer the question using a single word or a short phrase."
            message = incontext_msg + [ # TODO: add icl?
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question}
                ]}
            ]

        elif benchmark_info["task"] == 'caption':
            question = x['question']
            if isinstance(x.get("ground_truth"), str):
                x["ground_truth"] = [x["ground_truth"]] # need a list in evaluation

            incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)
            message = incontext_msg + [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question}
                ]}
            ]

        elif benchmark_info["task"] == 'region_caption':
            question = x['question']
            if isinstance(x.get("ground_truth"), str):
                x["ground_truth"] = [x["ground_truth"]] # need a list in evaluation

            incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)
            message = incontext_msg + [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"[identify] What is the object present at {question}"}
                ]}
            ]

        elif benchmark_info["task"] == "referring":
            question = x["question"]
            if prompt_type == 'json':
                message = [
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {
                            "type": "text", 
                            "text": f"Represent all {question} with quadrant bbox coordinates in a JSON format, represent objects with quadrant bbox: the corrdinates of each"
                    }
                    ]}
                ]
            elif prompt_type == 'plain':
                message = [
                    # {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": f"Find all {question}, delivering each oriented coordinates in plain text format 'x1,y1,x2,y2,x3,y3,x4,y4 object'"}
                    ]}
                ]
            else:
                raise ValueError(f"Unknown grounding prompt type: {prompt_type}")

        elif benchmark_info["task"] == "cls":
            classes_str = x["classes"]
            question = "Classify the image within one of the given classes:" + classes_str
            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question}
                ]}
            ]

        x["message"] = message
        x["idx"] = idx
    return data