import json
import os
import math
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from skimage.measure import block_reduce
import random


def select_hbb_json_prompt_template():
    MAIN_HBB_TEMPLATE = "Locate the {prompt}, output the bbox coordinates using JSON format"

    PROXY_HBB_TEMPLATE = [
        "Give me the bbox of {prompt} using JSON format",
        "Output the bbox of the {prompt} in the image using JSON format.",
        "From this image, provide the bbox for {prompt} using JSON format",
        "Please provide the bbox coordinate of the region this sentence describes: {prompt}",
        "Locate {prompt} in the given image and provide the bbox using JSON format"
    ]

    if random.random() < 0.5:
        template = MAIN_HBB_TEMPLATE
    else:
        template = random.choice(PROXY_HBB_TEMPLATE)

    if random.random() < 0.5:
        template = template.replace("bbox", "bounding box")

    return template
    

def select_obb_json_prompt_template():
    MAIN_OBB_TEMPLATE = "Locate the {prompt}, output the oriented bbox coordinates using JSON format"

    PROXY_OBB_TEMPLATE = [
        "Give me the oriented bounding box of {prompt} using JSON format",
        "Output the oriented bounding box of the {prompt} in the image using JSON format.",
        "From this image, provide the oriented bounding box for {prompt} using JSON format",
        "Please provide the oriented bounding box coordinate of the region this sentence describes: {prompt}",
        "Locate {prompt} in the given image and provide the oriented bounding box using JSON format"
    ]

    if random.random() < 0.5:
        template = MAIN_OBB_TEMPLATE
    else:
        template = random.choice(PROXY_OBB_TEMPLATE)

    if random.random() < 0.5:
        template = template.replace("bbox", "bounding box")

    if random.random() < 0.5:
        template = template.replace("oriented", "rotated")

    return template


def select_mask_prompt_template():
    MAIN_MASK_TEMPLATE = "Locate the {prompt}, output the segmentation mask"

    PROXY_MASK_TEMPLATE = [
        "Give me the segmentation mask of {prompt}",
        "Output the segmentation mask of the {prompt} in the image",
        "From this image, provide the segmentation mask for {prompt}",
        "Please provide the segmentation mask of the region this sentence describes: {prompt}",
        "Locate {prompt} in the given image and provide the segmentation mask", 
        "Segment the {prompt} in the given image"
    ]

    if random.random() < 0.5:
        template = MAIN_MASK_TEMPLATE
    else:
        template = random.choice(PROXY_MASK_TEMPLATE)

    return template


def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                encoded_row.append(f"{row[j-1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append(", ".join(encoded_row))
    return "; ".join(rows) + ";"


def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data


def get_hbb_conversations(meta):
    x1, y1, x2, y2 = [int(x) for x in meta['bbox']]
    question = meta['question']

    if random.random() < 0.5:
        # json format
        prompt_template = select_hbb_json_prompt_template()
        response = '```json\n[\n\t{"bbox_2d": [' + f"{x1}, {y1}, {x2}, {y2}" + '], "label": "' + question + '"}\n]\n```'

        conversations = [
            {"from": "human", "value": '<image>\n' + prompt_template.format(prompt=question)},
            {"from": "gpt", "value": response}, 
        ]  #  this llava-like conversation format need to be converted to gpt/qwen-like message format for apply_chat_template in the dataset/data_collator implementation.
    else:
        # plain text
        prompt_template = "find the {prompt}"
        response = f"{x1},{y1},{x2},{y2} {question}\n"

        conversations = [
            {"from": "system", "value": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
            {"from": "human", "value": '<image>\n' + prompt_template.format(prompt=question)},
            {"from": "gpt", "value": response}, 
        ]  #  this llava-like conversation format need to be converted to gpt/qwen-like message format for apply_chat_template in the dataset/data_collator implementation.
    return conversations


def get_obb_conversations(meta):
    # although hbb uses x1, y1, x2, y2, obb requires x, y, w, h, a
    bboxps = np.array(meta['poly']).astype(int)
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    if w < h:
        w, h = h, w
        a += 90
    while not 90 > a >= -90:
        if a >= 90:
            a -= 180
        else:
            a += 180
    assert 90 > a >= -90
    obb = [int(x), int(y), int(w), int(h), int(a)]

    question = meta['question']

    if random.random() < 0.5:
        # json format
        prompt_template = select_obb_json_prompt_template()
        response = '```json\n[\n\t{"oriented bbox": [' + f"{obb[0]}, {obb[1]}, {obb[2]}, {obb[3]}, {obb[4]}" + '], "label": "' + question + '"}\n]\n```'
        sys_prompt = "You are an AI assistant specializing in oriented object detection. Represent objects with: center (x,y), width (w), height (h), and rotation angle."
    else:
        # plain text
        prompt_template = "find the {prompt}"
        response = f"{obb[0]},{obb[1]},{obb[2]},{obb[3]},{obb[4]} {question}\n"
        sys_prompt = "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates in plain text format 'x,y,w,h,a object'."

    conversations = [
        {"from": "system", "value": sys_prompt},
        {"from": "human", "value": '<image>\n' + prompt_template.format(prompt=question)},
        {"from": "gpt", "value": response}, 
    ]  # this llava-like conversation format need to be converted to gpt/qwen-like message format for apply_chat_template in the dataset/data_collator implementation.
    return conversations


def normalize_polygon(polygon: list[int], clockwise: bool = True) -> list[tuple[int, int]]:
    """
    :param polygon: 输入为长度为8的list[int], 表示四边形的四个顶点 (x1, y1, x2, y2, x3, y3, x4, y4)。
    :param clockwise: 控制顶点排序的方向。True为顺时针, False为逆时针。
    """
    points = [(polygon[i], polygon[i + 1]) for i in range(0, 8, 2)]
    center = (sum(x for x, y in points) / 4, sum(y for x, y in points) / 4)
    angle_from_center = lambda point: math.atan2(point[1] - center[1], point[0] - center[0])
    points.sort(key=angle_from_center, reverse=not clockwise)
    start_index = points.index(min(points, key=lambda p: (p[1], p[0])))
    sorted_points = points[start_index:] + points[:start_index]
    return [x for p in sorted_points for x in p]


def get_polygon_example(meta):
    # x1, y1, x2, y2, x3, y3, x4, y4
    x1, y1, x2, y2, x3, y3, x4, y4 = normalize_polygon([int(x) for p in meta['poly'] for x in p], False)
    question = meta['question']

    if random.random() < 0.5:
        # json format
        prompt_template = select_obb_json_prompt_template()
        response = '```json\n[\n\t{"quadrant bbox": [' + f"{x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}" + '], "label": "' + question + '"}\n]\n```'
        sys_prompt = "You are an AI assistant specializing in oriented object detection. Represent objects with quadrant bbox: the corrdinates of each ."
    else:
        # plain text
        prompt_template = "find the {prompt}"
        response = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4} {question}\n"
        sys_prompt = "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates of the polygon vertices in plain text format 'x1,y1,x2,y2,x3,y3,x4,y4 object'."

    conversations = [
        {"from": "system", "value": sys_prompt},
        {"from": "human", "value": '<image>\n' + prompt_template.format(prompt=question)},
        {"from": "gpt", "value": response}, 
    ]  #  this llava-like conversation format need to be converted to gpt/qwen-like message format for apply_chat_template in the dataset/data_collator implementation.
    return conversations


def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                encoded_row.append(f"{row[j-1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append(", ".join(encoded_row))
    return "; ".join(rows) + ";"


def get_seg_conversations(meta, patch_size=16):
    mask_path = f"{mask_dir}/{meta['dataset']}/{meta['image_id'].split('.')[0] + '_' + str(meta['question_id']) + '.png'}"
    if not os.path.exists(mask_path):
        # if not ('geochat' in mask_path):
        #     import ipdb; ipdb.set_trace()
        return None

    mask_image = Image.open(mask_path).convert("L") 
    mask_array = np.array(mask_image)
    _, binary_mask = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)  # 二值化mask
    kernel = np.ones((50, 50), np.uint8)  # 定义形态学核（可以根据需求调整大小）
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # 应用闭运算来合并不相邻的区域
    contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓
    filled_mask = np.zeros_like(closed_mask)  # 创建一个空白的掩码，用于填充
    # 填充所有外部和内部轮廓 (如果该轮廓没有父轮廓，即为外部轮廓)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(filled_mask, contours, i, 1, cv2.FILLED)  # 填充外部轮廓及其内部区域

    filled_mask = Image.fromarray(filled_mask)
    original_size = filled_mask.size
    if original_size[0] >= original_size[1]:
        if original_size[0] % patch_size != 0:
            w_new = original_size[0] + (patch_size - original_size[0] % patch_size)
        else:
            w_new = original_size[0]
        h_new = w_new
    else:
        if original_size[1] % patch_size != 0:
            h_new = original_size[1] + (patch_size - original_size[1] % patch_size)
        else:
            h_new = original_size[1]
        w_new = h_new
    
    filled_mask = filled_mask.resize((w_new, h_new), Image.NEAREST)
    pooled_mask = block_reduce(np.array(filled_mask), block_size=(h_new // patch_size, w_new // patch_size), func=np.max)
    seg = encode_mask(pooled_mask)

    question = meta['question']

    prompt_template = select_mask_prompt_template()
    response = f"<seg>{seg}</seg>"  # XML format
    
    conversations = [
        {"from": "human", "value": '<image>\n' + prompt_template.format(prompt=question)},
        {"from": "gpt", "value": response}, 
    ]
    return conversations


if __name__ == "__main__":
    metainfo_file_list = [
        # "avvg_detection_train.jsonl",
        # "rrsisd_train.jsonl",
        "avvg_train.jsonl",
        "dior_rsvg_train.jsonl",
        "geochat_train.jsonl",
        "rsvg_train.jsonl",
        "vrsbench_train.jsonl"
    ]
    metainfo_dir = "playground/data/refGeo/metainfo"
    image_dir = "playground/data/refGeo/images"
    save_file = [
        "playground/data/refGeo/refgeo_mix_qwen2_5_vl_conversations.json", 
        "playground/data/refgeo_mix_qwen2_5_vl_conversations.json"
    ]
    mask_dir = "playground/data/refGeo/masks"

    data = []
    for metainfo_file in metainfo_file_list:
        examples = load_jsonl(os.path.join(metainfo_dir, metainfo_file))
        for example in examples:
            example["metainfo_filename"] = metainfo_file
        data.extend(examples)

    samples = []
    for _, meta in tqdm(enumerate(data), total=len(data)):
        image_id = meta['image_id']
        metainfo_filename = meta['metainfo_filename']

        if "geoground" in metainfo_filename:
            image_surfix = f"geoground/{meta['image_id']}"
        else:
            image_surfix = f"{meta['dataset']}/{meta['image_id']}"
        meta["image_surfix"] = image_surfix
        image_file_path = f"{image_dir}/{image_surfix}"
        meta["image_file_path"] = image_file_path

        for get_conversations in [get_hbb_conversations, get_obb_conversations, get_polygon_example, get_seg_conversations]:
            conversations = get_conversations(meta)
            if conversations is not None:
                sample = {
                    "id": len(samples), 
                    "image_id": image_id,
                    "image": meta["image_surfix"], 
                    "conversations": get_conversations(meta)
                }
            samples.append(sample)

    if isinstance(save_file, str):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w') as json_file:    
            json.dump(samples, json_file, indent=4)
    else:
        for file in save_file:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'w') as json_file:    
                json.dump(samples, json_file, indent=4)
