import re
import json
import random
from qwen_vl_utils import process_vision_info

from .sft_dataset import SupervisedDatasetForQwen2_5_VL


def has_box(text):
    return re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', text)


def are_all_boxes(text):
    return bool(re.match(r'^(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])*)\s*$', text.strip().rstrip('.')))


def compare_box_in_two_texts(text1, text2):
    bbox1 = re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', text1)
    bbox2 = re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', text2)
    return bbox1 and bbox2 and bbox1 == bbox2


def remove_box_from_text(text):
    return re.sub(r'\[(\d+(?:\s*,\s*\d+)*)\]\s', '', text)


def load_vision_and_get_size_from_messages(messages):
    image_inputs, video_inputs = process_vision_info(messages)

    if image_inputs is not None:
        size = image_inputs[0].size
    elif video_inputs is not None:
        size = video_inputs[0][0].size
    else:
        raise ValueError(f"No vision info in messages: {messages}")

    for message in messages:
        if isinstance(message["content"], list):
            for i, ele in enumerate(message["content"]):
                if "image" in ele or ele["type"] == "image":
                    message["content"][i]["ori_image"] = message["content"][i]["image"]
                    message["content"][i]["image"] = image_inputs.pop(0)
                if "image_url" in ele or ele["type"] == "image_url":
                    message["content"][i]["ori_image_url"] = message["content"][i].pop("image_url")
                    message["content"][i]["type"] = "image"
                    message["content"][i]["image"] = image_inputs.pop(0)
                if "video" in ele or ele["type"] == "video":
                    message["content"][i]["ori_video"] = message["content"][i]["video"]
                    message["content"][i]["video"] = video_inputs.pop(0)
    return size


def inv_normalize_coord(text, w, h, verbose=False):
    coord_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    
    def replace_coords(match):
        coords_str = match.group(1)
        coords = [int(x.strip()) for x in coords_str.split(',')]
        denormalized_coords = []
        for i in range(0, len(coords), 4):
            x1, y1, x2, y2 = coords[i:i+4]
            denorm_x1 = int(x1 * w / 100)
            denorm_y1 = int(y1 * h / 100)
            denorm_x2 = int(x2 * w / 100)
            denorm_y2 = int(y2 * h / 100)
            
            denormalized_coords.extend([denorm_x1, denorm_y1, denorm_x2, denorm_y2])
        return '[' + ', '.join(map(str, denormalized_coords)) + ']'
    result = re.sub(coord_pattern, replace_coords, text)
    if verbose:
        if text != result:
            print(f"{text=} -> {result=}")
    return result


def push_vision_setting_to_messages(
    messages, min_pixels, max_pixels, video_min_pixels=None, video_max_pixels=None,
):
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if (
                    "image" in ele or "image_url" in ele 
                    or ele["type"] in ("image", "image_url")
                ):
                    ele["min_pixels"] = min_pixels
                    ele["max_pixels"] = max_pixels
                if "video" in ele or ele["type"] == "video":
                    if video_min_pixels is not None:
                        ele["min_pixels"] = video_min_pixels
                    if video_max_pixels is not None:
                        ele["max_pixels"] = video_max_pixels


def grounding_format_teochat_to_qwen25vl(text, prompt, use_json_prompt=False):
    category_name = (
        prompt
        .replace("This is a satellite image:", "")
        .replace("<video>", "").replace("<image>", "")
        .replace("Identify", "")
        .replace("the location of", "")
        .replace("Include bounding boxes of the form [x_min, y_min, x_max, y_max] for each identified object in your response", "")
        .replace("Include a bounding box of the form [x_min, y_min, x_max, y_max] for each identified building in your response.", "")
        .replace("If there are no such buildings, do not output a bounding box", "")
        .replace("What is ", "")
        .translate(str.maketrans('', '', '0123456789'))
        # .replace("the ", "")
        .replace("all ", "")
        .replace("  ", " ")
        .replace("?", "")
        .strip(".").strip().strip(".").strip()
    )

    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, text)

    boxes = []
    for match in matches:
        box = [int(coord) for coord in match]
        boxes.append(box)

    if use_json_prompt:
        response = (
            '```json\n[\n\t' +
            ',\n\t'.join(json.dumps({"bbox_2d": box, "label": category_name}) for box in boxes)
            + '\n]\n```'
        )
    else:
        response = (
            '\n'.join(f'{box[0]},{box[1]},{box[2]},{box[3]} {category_name}' for box in boxes)
            + '\n'
        )
    return response


def process_teochatlas_messages(messages, min_pixels, max_pixels, video_min_pixels=None, video_max_pixels=None, use_json_prompt=False):
    size = None
    def _process(text):
        nonlocal size
        if has_box(text):
            if size is None:
                push_vision_setting_to_messages(messages, min_pixels, max_pixels, video_min_pixels, video_max_pixels)
                size = load_vision_and_get_size_from_messages(messages)
            w, h = size
            return inv_normalize_coord(text, w, h)
        else:
            return text

    for msg in messages:
        # inverse normalize the box (with resized xxx)
        content = msg['content']
        if isinstance(content, str):
            msg['content'] = _process(msg['content'])
        elif isinstance(content, list):
            for i in range(len(content)):
                if isinstance(content[i], str):
                    content[i] = [{
                        "type": "text",
                        "text": _process(content[i]),
                        "ori_text": content[i],
                    }]
                elif isinstance(content[i], dict) and content[i]['type'] == 'text':
                    new_text = _process(content[i]['text'])
                    if content[i]['text'] != new_text:
                        content[i]['ori_text'] = content[i]['text']
                        content[i]['text'] = new_text

        # delete dirty strings in prompt
        if msg['role'] == 'user':
            if isinstance(msg['content'], str):
                user_text = msg['content'].replace(". ?", ".").replace(".. ", ".").strip()
                msg['content'] = user_text
            elif isinstance(msg['content'], list):
                for content in msg['content']:
                    if content['type'] == 'text':
                        user_text = content['text'].replace(". ?", ".").replace(".. ", ".").strip()
                        content['text'] = user_text
            else:
                raise ValueError(f"Unknown user message: {msg['content']}")
    
        # when the sample is for REC/grounding task, refactor the response to Qwen2.5-VL format (this type of data in TeoChat always detect building)
        if msg['role'] == 'assistant':
            if isinstance(msg['content'], str):
                if are_all_boxes(msg['content']):
                    msg['content'] = [{
                            "type": "text",
                            "text": grounding_format_teochat_to_qwen25vl(msg['content'], user_text, use_json_prompt),
                            "ori_text": msg['content'], 
                    }]
            elif isinstance(msg['content'], list):
                for content in msg['content']:
                    if content['type'] == 'text':
                        if are_all_boxes(content['text']):
                            if 'ori_text' not in content:
                                content['ori_text'] = content['text']
                            else:
                                content['ori_text'] = content['ori_text'] + " -> " + content['text']
                            content['text'] = grounding_format_teochat_to_qwen25vl(content['text'], user_text, use_json_prompt)
    return messages


class TeoChatDatasetForQwen2_5_VL(SupervisedDatasetForQwen2_5_VL):
    """Dataset supporting GeoChat and TeoChat (Adapting box)"""

    def __init__(self, *args, **kwargs):
        super(TeoChatDatasetForQwen2_5_VL, self).__init__(*args, **kwargs)
        
        # for region_captioning task, there is repeated bbox in the response, we need to remove them
        for example in self.list_data_meta:
            # if example["task"].endswith("region_captioning"):
            if compare_box_in_two_texts(example['conversations'][0]['value'], example['conversations'][1]['value']):
                example['conversations'][1]['ori_value'] = example['conversations'][1]['value']
                example['conversations'][1]['value'] = remove_box_from_text(example['conversations'][1]['value'])

    def get_messages(self, idx):
        messages = super(TeoChatDatasetForQwen2_5_VL, self).get_messages(idx)
        return process_teochatlas_messages(
            messages, self.min_pixels, self.max_pixels, self.video_min_pixels, self.video_max_pixels, 
            use_json_prompt=random.random() >= getattr(self.data_args, "prob_plain_text_prompt", 1.0)
        )
