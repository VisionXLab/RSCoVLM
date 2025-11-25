import os
import cv2
import math
import random
import logging
from tqdm import tqdm
from copy import deepcopy
from typing import Optional

import numpy as np
from PIL import Image
from skimage.measure import block_reduce

from torch.utils.data import Dataset
import transformers

from qwen_vl_utils import smart_resize
from ..params import DataArguments
from .processing_message import process_to_train_qwen2_5_vl_with_default_chat_template
from rscovlm.utils import load_json_or_jsonl, random_sample


IGNORE_INDEX = -100


def select_hbb_json_prompt_template(prob_proxy_prompt=0.5):
    MAIN_HBB_TEMPLATE = "Locate the {prompt}, output the bbox coordinates using JSON format"

    PROXY_HBB_TEMPLATE = [
        "Give me the bbox of {prompt} using JSON format",
        "Output the bbox of the {prompt} in the image using JSON format.",
        "From this image, provide the bbox for {prompt} using JSON format",
        "Please provide the bbox coordinate of the region this sentence describes: {prompt}",
        "Locate {prompt} in the given image and provide the bbox using JSON format"
    ]

    if random.random() < prob_proxy_prompt:
        template = random.choice(PROXY_HBB_TEMPLATE)
    else:
        template = MAIN_HBB_TEMPLATE

    if random.random() < prob_proxy_prompt:
        template = template.replace("bbox", "bounding box")

    return template
    

def select_obb_json_prompt_template(prob_proxy_prompt=0.5):
    MAIN_OBB_TEMPLATE = "Locate the {prompt}, output the oriented bbox coordinates using JSON format"

    PROXY_OBB_TEMPLATE = [
        "Give me the oriented bounding box of {prompt} using JSON format",
        "Output the oriented bounding box of the {prompt} in the image using JSON format.",
        "From this image, provide the oriented bounding box for {prompt} using JSON format",
        "Please provide the oriented bounding box coordinate of the region this sentence describes: {prompt}",
        "Locate {prompt} in the given image and provide the oriented bounding box using JSON format"
    ]

    if random.random() < prob_proxy_prompt:
        template = random.choice(PROXY_OBB_TEMPLATE)
    else:
        template = MAIN_OBB_TEMPLATE

    if random.random() < prob_proxy_prompt:
        template = template.replace("bbox", "bounding box")

    if random.random() < prob_proxy_prompt:
        template = template.replace("oriented", "rotated")

    return template


def select_mask_prompt_template(prob_proxy_prompt=0.5):
    MAIN_MASK_TEMPLATE = "Locate the {prompt}, output the segmentation mask"

    PROXY_MASK_TEMPLATE = [
        "Give me the segmentation mask of {prompt}",
        "Output the segmentation mask of the {prompt} in the image",
        "From this image, provide the segmentation mask for {prompt}",
        "Please provide the segmentation mask of the region this sentence describes: {prompt}",  # TODO: maybe this should be main
        "Locate {prompt} in the given image and provide the segmentation mask", 
        "Segment the {prompt} in the given image"
    ]

    if random.random() < prob_proxy_prompt:
        template = random.choice(PROXY_MASK_TEMPLATE)
    else:
        template = MAIN_MASK_TEMPLATE

    return template


def get_hbb_messages(meta, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    x1, y1, x2, y2 = [int(x) for x in meta['bbox']]
    question = meta['question']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}

    if 'resized_width' in meta or 'resized_height' in meta:
        x1 = int(x1 * meta['resized_width'] / meta['image_width'])
        y1 = int(y1 * meta['resized_height'] / meta['image_height'])
        x2 = int(x2 * meta['resized_width'] / meta['image_width'])
        y2 = int(y2 * meta['resized_height'] / meta['image_height'])

        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']

    if random.random() > prob_plain_text_prompt:
        # json format
        prompt_template = select_hbb_json_prompt_template(prob_proxy_prompt)
        response = '```json\n[\n\t{"bbox_2d": [' + f"{x1}, {y1}, {x2}, {y2}" + '], "label": "' + question + '"}\n]\n```'

        messages = [
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": prompt_template.format(prompt=question)}
            ]},
            {"role": "assistant", "content": response}, 
        ]
    else:
        # plain text
        prompt_template = "find the {prompt}"
        response = f"{x1},{y1},{x2},{y2} {question}\n"

        messages = [
            {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": prompt_template.format(prompt=question)}
            ]},
            {"role": "assistant", "content": response}, 
        ]
    return messages


def get_obb_messages(meta, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    # although hbb uses x1, y1, x2, y2, obb requires x, y, w, h, a
    bboxps = np.array(meta['poly']).astype(int)
    question = meta['question']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}

    if 'resized_width' in meta or 'resized_height' in meta:
        bboxps[:, 0] = (bboxps[:, 0] * meta['resized_width'] / meta['image_width']).astype(int)
        bboxps[:, 1] = (bboxps[:, 1] * meta['resized_height'] / meta['image_height']).astype(int)

        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']

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

    if random.random() > prob_plain_text_prompt:
        # json format
        prompt_template = select_obb_json_prompt_template(prob_proxy_prompt)
        response = '```json\n[\n\t{"oriented bbox": [' + f"{obb[0]}, {obb[1]}, {obb[2]}, {obb[3]}, {obb[4]}" + '], "label": "' + question + '"}\n]\n```'
        sys_prompt = "You are an AI assistant specializing in oriented object detection. Represent objects with: center (x,y), width (w), height (h), and rotation angle."
    else:
        # plain text
        prompt_template = "find the {prompt}"
        response = f"{obb[0]},{obb[1]},{obb[2]},{obb[3]},{obb[4]} {question}\n"
        sys_prompt = "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates in plain text format 'x,y,w,h,a object'."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            image_content,
            {"type": "text", "text": prompt_template.format(prompt=question)}
        ]},
        {"role": "assistant", "content": response}, 
    ]
    return messages


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


def get_polygon_messages(meta, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    # x1, y1, x2, y2, x3, y3, x4, y4
    polygon = meta['poly']
    question = meta['question']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}

    if 'resized_width' in meta or 'resized_height' in meta:
        polygon = np.array(polygon).astype(int)
        polygon[:, 0] = (polygon[:, 0] * meta['resized_width'] / meta['image_width']).astype(int)
        polygon[:, 1] = (polygon[:, 1] * meta['resized_height'] / meta['image_height']).astype(int)
        polygon = polygon.tolist()

        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']

    x1, y1, x2, y2, x3, y3, x4, y4 = normalize_polygon([int(x) for p in polygon for x in p], False)

    if random.random() > prob_plain_text_prompt:
        # json format
        prompt_template = select_obb_json_prompt_template(prob_proxy_prompt)
        response = '```json\n[\n\t{"quadrant bbox": [' + f"{x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}" + '], "label": "' + question + '"}\n]\n```'
        sys_prompt = "You are an AI assistant specializing in oriented object detection. Represent objects with quadrant bbox: the corrdinates of each vertices of the oriented bounding boxes in clock-wise order."
    else:
        # plain text
        prompt_template = "find the {prompt}"
        response = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4} {question}\n"
        sys_prompt = "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates of the polygon vertices in plain text format 'x1,y1,x2,y2,x3,y3,x4,y4 object'."
        
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            image_content,
            {"type": "text", "text": prompt_template.format(prompt=question)}
        ]},
        {"role": "assistant", "content": response}, 
    ]
    return messages


def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                encoded_row.append(f"{row[j-1]}*{count}")
                count = 1
        encoded_row.append(f"{row[-1]}*{count}")
        rows.append(",".join(encoded_row))
    return ";".join(rows) + ";"


def get_seg_messages(meta, patch_size=28, prob_proxy_prompt=0.5):
    mask_image = Image.open(meta["mask_file_path"]).convert("L") 
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

    mask_width, mask_height = filled_mask.size
    assert mask_width == meta['image_width'] and mask_height == meta['image_height'], f"Mask size {mask_width}x{mask_height} does not match image size {meta['image_width']}x{meta['image_height']}"

    filled_mask = filled_mask.resize((meta['resized_width'], meta['resized_height']), Image.NEAREST)  # TODO: resized_width should in if
    pooled_mask = block_reduce(np.array(filled_mask), block_size=(patch_size, patch_size), func=np.max)
    seg = encode_mask(pooled_mask)

    question = meta['question']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}
    if 'resized_width' in meta or 'resized_height' in meta:
        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']

    prompt_template = select_mask_prompt_template(prob_proxy_prompt)
    response = f"<seg>{seg}</seg>"  # XML format
    sys_prompt = "As an AI assistant, you specialize in segmentation, delivering segmentation results as Row-wise RLE format. Each row is represented as '<value>*<count>', and rows are separated by ';'. The last row ends with a semicolon. For example, '0*1,1*2;0*3;' represents the mask."  # TODO: refine this prompt
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            image_content,
            {"type": "text", "text": prompt_template.format(prompt=question)}
        ]},
        {"role": "assistant", "content": response}, 
    ]
    return messages


def get_messages(region_type, meta, patch_size, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    if region_type == 'hbb':
        return get_hbb_messages(meta, prob_proxy_prompt, prob_plain_text_prompt)
    elif region_type == 'obb':
        return get_obb_messages(meta, prob_proxy_prompt, prob_plain_text_prompt)
    elif region_type == 'polygon':
        return get_polygon_messages(meta, prob_proxy_prompt, prob_plain_text_prompt)
    elif region_type == 'seg':
        return get_seg_messages(meta, patch_size, prob_proxy_prompt)  # TODO: upgrade patch_size to be dynamic
    else:
        raise ValueError(f"Unknown region type: {region_type}")


class GeoGroundDatasetForQwen2_5_VL(Dataset):
    # TODO: considering supporting dynamic resolution and packed dataset
    region_types = ['hbb', 'obb', 'polygon', 'seg']

    def __init__(
        self, 
        data_config_list: list[dict],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_for_position_ids: Optional[transformers.PreTrainedModel] = None,
    ):
        list_data_dict = []
        for i, data in enumerate(data_config_list):
            annotations = load_json_or_jsonl(data["annotation_path"]) # anno contents
            annotations = random_sample(data, annotations, getattr(data_args, 'data_sampling_seed', None))
            logging.info(f"sampling {len(annotations)} examples from {data_config_list[i]}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations
        
        self.list_data_dict = list_data_dict
        self.processor = processor
        self.data_args = data_args
        self.model_for_position_ids = model_for_position_ids

        self.min_pixels = getattr(self.data_args, 'min_pixels', processor.image_processor.min_pixels)
        self.max_pixels = getattr(self.data_args, 'max_pixels', processor.image_processor.max_pixels)
        self.patch_size = self.processor.image_processor.patch_size
        self.merge_size = self.processor.image_processor.merge_size
        self.image_factor = self.patch_size * self.merge_size
        self.prob_random_resize = getattr(self.data_args, 'prob_random_resize', 0.0)
        self.prob_proxy_prompt = getattr(self.data_args, 'prob_proxy_prompt', 0.5)
        self.prob_plain_text_prompt = getattr(self.data_args, 'prob_plain_text_prompt', 0.5)

        for meta in self.list_data_dict:
            meta["image_surfix"] = f"{meta['dataset']}/{meta['image_id']}"
            meta["image_file_path"] = os.path.join(meta["data_path"], 'images', meta["dataset"], meta['image_id'])

        todo_list = [
            (metainfo_id, region_type)
            for region_type in self.region_types if region_type != 'seg'
            for metainfo_id in range(len(self.list_data_dict))
        ]

        if 'seg' in self.region_types:
            # check how many segmentation masks are available
            for metainfo_id in tqdm(range(len(self.list_data_dict)), desc="Checking segmentation masks exist"):  # , disable=(torch.distributed.get_rank() > 0)
                meta = self.list_data_dict[metainfo_id]
                mask_root = os.path.join(meta['data_path'], "masks")
                mask_path = f"{mask_root}/{meta['dataset']}/{meta['image_id'].split('.')[0] + '_' + str(meta['question_id']) + '.png'}"
                if 'mask_exists' in meta:
                    mask_exists = meta["mask_exists"]
                else:
                    mask_exists = os.path.exists(mask_path)
                meta["mask_file_path"] = mask_path if mask_exists else None
                if mask_exists:
                    todo_list.append((metainfo_id, 'seg'))

        # sort by metainfo_id, then by region_type
        todo_list = sorted(todo_list, key=lambda x: (x[0], self.region_types.index(x[1])))
        self.todo_list = todo_list
        
    def add_resized_size_to_metainfo(self, meta):
        original_width, original_height = meta["image_width"], meta["image_height"]

        resized_height, resized_width = smart_resize(
            original_height, original_width, factor=self.image_factor,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        
        meta["resized_width"] = resized_width
        meta["resized_height"] = resized_height
        return meta
    
    def __len__(self):
        return len(self.todo_list)

    def random_resize(self, meta):
        original_width, original_height = meta["image_width"], meta["image_height"]
        original_pixels = original_width * original_height
        scale = random.uniform(self.min_pixels / original_pixels, self.max_pixels / original_pixels)
        resized_height, resized_width = int(original_height * scale), int(original_width * scale)
        resized_height, resized_width = smart_resize(
            resized_height, resized_width, factor=self.image_factor,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        meta["resized_width"] = resized_width
        meta["resized_height"] = resized_height
        return meta

    def get_messages(self, idx):
        metainfo_id, region_type = self.todo_list[idx]
        meta = deepcopy(self.list_data_dict[metainfo_id])
        if random.random() <= self.prob_random_resize:
            self.random_resize(meta)
        else:
            self.add_resized_size_to_metainfo(meta)
        return get_messages(region_type, meta, self.image_factor, self.prob_proxy_prompt, self.prob_plain_text_prompt)

    def preprocess(self, messages,**kwargs):
        return process_to_train_qwen2_5_vl_with_default_chat_template(
            self.processor, messages, 
            min_pixels=self.min_pixels, 
            max_pixels=self.max_pixels,
            model_for_position_ids=self.model_for_position_ids,
        )
                
    def __getitem__(self, idx):
        messages = self.get_messages(idx)
        return self.preprocess(messages)


class GeoGroundSegOnlyDatasetForQwen2_5_VL(GeoGroundDatasetForQwen2_5_VL):
    region_types = ['seg']


class GeoGroundHbbOnlyDatasetForQwen2_5_VL(GeoGroundDatasetForQwen2_5_VL):
    region_types = ['hbb']


class GeoGroundPolyOnlyDatasetForQwen2_5_VL(GeoGroundDatasetForQwen2_5_VL):
    region_types = ['polygon']


class GeoGroundWoSegDatasetForQwen2_5_VL(GeoGroundDatasetForQwen2_5_VL):
    region_types = ['hbb', 'obb', 'polygon']