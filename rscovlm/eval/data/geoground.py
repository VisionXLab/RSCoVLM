import os
import json


GEOGROUND = {
    "avvg_test": {
        "image_prefix": "images/avvg",
        "ann_filename": "avvg_test.jsonl",
    }, 
    "dior_rsvg_val": {
        "image_prefix": "images/dior_rsvg",
        "ann_filename": "dior_rsvg_val.jsonl",
    },
    "dior_rsvg_test": {
        "image_prefix": "images/dior_rsvg",
        "ann_filename": "dior_rsvg_test.jsonl",
    },
    "rsvg_test": {
        "image_prefix": "images/rsvg",
        "ann_filename": "rsvg_test.jsonl",
    },
    "rsvg_val": {
        "image_prefix": "images/rsvg",
        "ann_filename": "rsvg_val.jsonl",
    },
    "geochat_test": {
        "image_prefix": "images/geochat_test",
        "ann_filename": "geochat_test.jsonl",
    },
    "vrsbench_test": {
        "image_prefix": "images/vrsbench",
        "ann_filename": "vrsbench_test.jsonl",
    }
}


def _prepare_geoground_eval(data_root, image_prefix, ann_filename, grounding_prompt_type, resize_to=None):
    ann_filepath = os.path.join(data_root, 'metainfo', ann_filename)

    with open(ann_filepath, 'r') as f:
        data = [json.loads(line) for line in f]

    for idx, x in enumerate(data):
        image_name = x['image_id']
        image_path = os.path.join(data_root, image_prefix, image_name)

        if "geochat" in image_path:
            if not os.path.exists(image_path):
                image_path = image_path.replace(".jpg", ".png")
        assert os.path.exists(image_path), f"Image path {image_path} does not exist"

        image_content = {"type": "image", "image": f"file://{image_path}"}
        if resize_to is not None:
            image_content["min_pixels"] = resize_to ** 2
            image_content["max_pixels"] = resize_to ** 2

        if grounding_prompt_type == 'json':
            message = [
                {"role": "user", "content": [
                    image_content,
                    {"type": "text", "text": f"Locate the {x['question']}, output the bbox coordinates using JSON format"}
                ]}
            ]
        elif grounding_prompt_type == 'plain':
            message = [
                {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
                {"role": "user", "content": [
                    image_content,
                    {"type": "text", "text": f"find the {x['question']}"}
                ]}
            ]
        else:
            raise ValueError(f"Unknown grounding prompt type: {grounding_prompt_type}")

        x["message"] = message
        x["idx"] = idx
    return data


def prepare_geoground_eval(data_root, benchmark, grounding_prompt_type):
    resize_to = None
    if "_336" in benchmark:
        resize_to = 336
        benchmark = benchmark.replace("_336", "")
    elif "_224" in benchmark:
        resize_to = 224
        benchmark = benchmark.replace("_224", "")
    elif "_448" in benchmark:
        resize_to = 448
        benchmark = benchmark.replace("_448", "")
    elif "_1008" in benchmark:
        resize_to = 1008
        benchmark = benchmark.replace("_1008", "")

    if benchmark in ["geoground_all", "geoground"]:
        data_pipe_list = []
        for dataset_name, kwargs in GEOGROUND.items():
            data = _prepare_geoground_eval(data_root, **kwargs, grounding_prompt_type=grounding_prompt_type, resize_to=resize_to)
            for sample in data:
                sample["dataset_name"] = dataset_name
            data_pipe_list.extend(data)

    else:
        benchmark = benchmark.lstrip("geoground_")
        kwargs = GEOGROUND[benchmark]
        data = _prepare_geoground_eval(data_root, **kwargs, grounding_prompt_type=grounding_prompt_type, resize_to=resize_to)
        for sample in data:
            sample["dataset_name"] = benchmark
        data_pipe_list = data
    return data_pipe_list