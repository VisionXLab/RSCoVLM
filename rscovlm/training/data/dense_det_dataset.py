import os
import re
import cv2
import math
import json
import random
import logging
from tqdm import tqdm
from copy import deepcopy
from typing import Optional

import numpy as np
from PIL import Image
from skimage.measure import block_reduce

import torch.distributed as dist
from torch.utils.data import Dataset, get_worker_info

import transformers

from qwen_vl_utils import smart_resize
from ..params import DataArguments
from .processing_message import process_to_train_qwen2_5_vl_with_default_chat_template
from rscovlm.utils import load_json_or_jsonl, random_sample

IGNORE_INDEX = -100


def preprocess_coco_format_metadata(metadata, keep_empty_gt=True):
    images = metadata['images']
    annotations = metadata['annotations']
    categories = metadata['categories']

    categoryid2rank = {}
    categoryid2name = {}
    cat_names = sorted([_["name"] for _ in categories], key=lambda s: s.replace("-", " ").lower())
    for cat in categories:
        category_id = cat["id"]
        category_name = cat["name"]
        categoryid2rank[category_id] = cat_names.index(category_name)
        categoryid2name[category_id] = category_name.replace("-", " ").lower()
    categories_str = ', '.join([c.replace("-", " ").lower() for c in cat_names])

    imgid2data = {}
    for image in images:
        imgid = image['id']
        image['image_width'] = image.pop('width')
        image['image_height'] = image.pop('height')
        image['categories_str'] = categories_str
        imgid2data[imgid] = image
        imgid2data[imgid]['annotations'] = []

    for ann in annotations:
        imgid = ann['image_id']
        catid = ann['category_id']

        # coco format -> x1y1x2y2
        x1, y1, w, h = ann['bbox']
        x2, y2 = int(x1 + w), int(y1 + h)
        ann['bbox'] = [x1, y1, x2, y2]

        ann['category_name'] = categoryid2name[catid]
        ann['category_rank'] = categoryid2rank[catid]
        imgid2data[imgid]['annotations'].append(ann)

    if keep_empty_gt:
        return list(imgid2data.values())
    else:
        return [item for item in imgid2data.values() if len(item['annotations']) > 0]


def select_hbb_json_prompt_template(prob_proxy_prompt=0.5):
    MAIN_HBB_TEMPLATE = "Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes {categories}."  # follow PR1

    PROXY_HBB_TEMPLATE = [
        "Please output bbox coordinates and names of {categories} in JSON format."

        "Give me the bboxes and names of {categories} using JSON format",
        "Output the category names and bboxes of the {categories} in the image using JSON format.",
        "From this image, provide the bboxes and categories for {categories} using JSON format",
        "Please provide the bbox coordinates of the following objects: {categories}.",
        "Locate {categories} in the given image and provide the bboxes using JSON format"
    ]

    if random.random() <= prob_proxy_prompt:
        template = random.choice(PROXY_HBB_TEMPLATE)
    else:
        template = MAIN_HBB_TEMPLATE

    if random.random() <= prob_proxy_prompt:
        template = template.replace("bbox", "bounding box")

    return template
    

def select_obb_json_prompt_template(prob_proxy_prompt=0.5):
    MAIN_OBB_TEMPLATE = "Locate every item from the category list in the image and output the oriented bbox coordinates in JSON format. The category set includes {categories}."  # follow PR1

    PROXY_OBB_TEMPLATE = [
        "Give me the oriented bboxes and names of {categories} using JSON format",
        "Output the category names and oriented bboxes of the {categories} in the image using JSON format.",
        "From this image, provide the oriented bboxes and categories for {categories} using JSON format",
        "Please provide the oriented bbox coordinates of the following objects: {categories}",
        "Locate {categories} in the given image and provide the oriented bboxes using JSON format"
    ]

    if random.random() <= prob_proxy_prompt:
        template = random.choice(PROXY_OBB_TEMPLATE)
    else:
        template = MAIN_OBB_TEMPLATE

    if random.random() <= prob_proxy_prompt:
        template = template.replace("bbox", "bounding box")

    if random.random() <= prob_proxy_prompt:
        template = template.replace("oriented", "rotated")

    return template


def resize_coordinates_in_metadata(annotations, ori_width, ori_height, resized_width, resized_height):
    if isinstance(annotations, dict):
        annotations = [annotations]

    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        x1 = int(x1 * resized_width / ori_width)
        y1 = int(y1 * resized_height / ori_height)
        x2 = int(x2 * resized_width / ori_width)
        y2 = int(y2 * resized_height / ori_height)
        ann['bbox'] = [x1, y1, x2, y2]

        if 'segmentation' in ann:
            seg = ann['segmentation']
            assert len(seg) == 1, ann
            assert len(seg[0]) % 2 == 0, ann

            for i in range(0, len(seg[0]), 2):
                seg[0][i] = int(seg[0][i] * resized_width / ori_width)
                seg[0][i + 1] = int(seg[0][i + 1] * resized_height / ori_height)

            ann['segmentation'] = seg
    return annotations

def set_open_end_prompt_settings(template, annotations, synonyms_dict):
    OPEN_END_TEMPLATE = 'all the visible remote sensing objects'
    template = re.sub(r'(the )?{categories}', OPEN_END_TEMPLATE, template)
    for ann in annotations:
        regular_category_name = re.sub(r"[_\-]+", " ", ann['category_name']).lower()
        ann['category_name'] = random.choice(synonyms_dict[regular_category_name])
    return template, annotations


def get_hbb_messages(meta, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    annotations = meta['annotations']
    categories = meta['categories_str']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}
    synonyms_dict = meta.get("synonyms_dict", None)
    use_synonyms = meta.get("use_synonyms", False)

    if 'resized_width' in meta or 'resized_height' in meta:
        annotations = resize_coordinates_in_metadata(annotations, meta['image_width'], meta['image_height'], meta['resized_width'], meta['resized_height'])
        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']
    annotations.sort(key=lambda ann: (ann["category_rank"], ann["bbox"][1], ann["bbox"][0]))

    if random.random() >= prob_plain_text_prompt:
        # json format
        prompt_template = select_hbb_json_prompt_template(prob_proxy_prompt)

        if use_synonyms: 
            prompt_template, annotations = set_open_end_prompt_settings(prompt_template, annotations, synonyms_dict)
        
        if len(annotations) == 0:
            response = 'There are none.'
        else:
            response = (
                '```json\n[\n\t' +
                ',\n\t'.join(json.dumps({"bbox_2d": ann["bbox"], "label": ann["category_name"]}) for ann in annotations)
                + '\n]\n```'
            )  # TODO: consider what to response for images with no objects
        messages = [
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": prompt_template.format(categories=categories)}
            ]},
            {"role": "assistant", "content": response}, 
        ]
    else:
        # plain text
        prompt_template = "find the {categories}"

        if use_synonyms: 
            prompt_template, annotations = set_open_end_prompt_settings(prompt_template, annotations, synonyms_dict)
        
        if len(annotations) == 0:
            response = 'There are none.'
        else:
            response = (
                '\n'.join(f'{ann["bbox"][0]},{ann["bbox"][1]},{ann["bbox"][2]},{ann["bbox"][3]} {ann["category_name"]}' for ann in annotations)
                + '\n'
            )  # TODO: consider what to response for images with no objects

        messages = [
            {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": prompt_template.format(categories=categories)}
            ]},
            {"role": "assistant", "content": response}, 
        ]
    return messages


def convert_segmentation_to_obb(segmentation):
    assert len(segmentation) == 1, segmentation
    polygon = segmentation[0]
    bboxps = np.array(polygon).astype(int).reshape(-1, 2)
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
    return obb


def get_obb_messages(meta, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    # although hbb uses x1, y1, x2, y2, obb requires x, y, w, h, a
    annotations = meta['annotations']
    categories = meta['categories_str']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}
    synonyms_dict = meta.get("synonyms_dict", None)
    use_synonyms = meta.get("use_synonyms", False)

    if 'resized_width' in meta or 'resized_height' in meta:
        annotations = resize_coordinates_in_metadata(annotations, meta['image_width'], meta['image_height'], meta['resized_width'], meta['resized_height'])
        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']
    annotations.sort(key=lambda ann: (ann["category_rank"], ann["bbox"][1], ann["bbox"][0]))

    if random.random() >= prob_plain_text_prompt:
        # json format
        prompt_template = select_obb_json_prompt_template(prob_proxy_prompt)

        if use_synonyms: 
            prompt_template, annotations = set_open_end_prompt_settings(prompt_template, annotations, synonyms_dict)
        
        if len(annotations) == 0:
            response = 'There are none.'
        else:
            response = []
            for ann in annotations:
                obb = convert_segmentation_to_obb(ann["segmentation"])
                response.append(json.dumps({"oriented bbox": obb, "label": ann["category_name"]}))
            response = '```json\n[\n\t' + ',\n\t'.join(response) + '\n]\n```'  # TODO: consider what to response for images with no objects
        sys_prompt = "You are an AI assistant specializing in oriented object detection. Represent objects with: center (x,y), width (w), height (h), and rotation angle."
    else:
        # plain text
        prompt_template = "find the {categories}"

        if use_synonyms: 
            prompt_template, annotations = set_open_end_prompt_settings(prompt_template, annotations, synonyms_dict)
        
        if len(annotations) == 0:
            response = 'There are none.'
        else:
            response = []
            for ann in annotations:
                x, y, w, h, a = convert_segmentation_to_obb(ann["segmentation"])
                response.append(f'{x},{y},{w},{h},{a} {ann["category_name"]}')
            response = '\n'.join(response) + '\n'  # TODO: consider what to response for images with no objects
        sys_prompt = "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates in plain text format 'x,y,w,h,a object'."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            image_content,
            {"type": "text", "text": prompt_template.format(categories=categories)}
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
    annotations = meta['annotations']
    categories = meta['categories_str']
    image_file_path = meta["image_file_path"]
    image_content = {"type": "image", "image": f"file://{image_file_path}"}
    synonyms_dict = meta.get("synonyms_dict", None)
    use_synonyms = meta.get("use_synonyms", False)

    if 'resized_width' in meta or 'resized_height' in meta:
        annotations = resize_coordinates_in_metadata(annotations, meta['image_width'], meta['image_height'], meta['resized_width'], meta['resized_height'])
        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']
    annotations.sort(key=lambda ann: (ann["category_rank"], ann["bbox"][1], ann["bbox"][0]))

    if random.random() >= prob_plain_text_prompt:
        # json format
        prompt_template = select_obb_json_prompt_template(prob_proxy_prompt)

        if use_synonyms: 
            prompt_template, annotations = set_open_end_prompt_settings(prompt_template, annotations, synonyms_dict)
        
        if len(annotations) == 0:
            response = 'There are none.'
        else:
            response = []
            for ann in annotations:
                assert len(ann["segmentation"]) == 1, ann
                polygon = normalize_polygon(ann["segmentation"][0])
                response.append(json.dumps({"quadrant bbox": polygon, "label": ann["category_name"]}))
            response = '```json\n[\n\t' + ',\n\t'.join(response) + '\n]\n```'  # TODO: consider what to response for images with no objects
        sys_prompt = "You are an AI assistant specializing in oriented object detection. Represent objects with quadrant bbox: the corrdinates of each vertices of the oriented bounding boxes in clock-wise order."
    else:
        # plain text
        prompt_template = "find the {categories}"

        if use_synonyms: 
            prompt_template, annotations = set_open_end_prompt_settings(prompt_template, annotations, synonyms_dict)
        
        if len(annotations) == 0:
            response = 'There are none.'
        else:
            response = []
            for ann in annotations:
                assert len(ann["segmentation"]) == 1, ann
                x1, y1, x2, y2, x3, y3, x4, y4 = normalize_polygon(ann["segmentation"][0])
                response.append(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4} {ann["category_name"]}')
            response = '\n'.join(response) + '\n'  # TODO: consider what to response for images with no objects
        sys_prompt = "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates of the polygon vertices in plain text format 'x1,y1,x2,y2,x3,y3,x4,y4 object'."
        
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            image_content,
            {"type": "text", "text": prompt_template.format(categories=categories)}
        ]},
        {"role": "assistant", "content": response}, 
    ]
    return messages


def get_messages(region_type, meta, prob_proxy_prompt=0.5, prob_plain_text_prompt=0.5):
    if region_type == 'hbb':
        return get_hbb_messages(meta, prob_proxy_prompt, prob_plain_text_prompt)
    elif region_type == 'obb':
        return get_obb_messages(meta, prob_proxy_prompt, prob_plain_text_prompt)
    elif region_type == 'polygon':
        return get_polygon_messages(meta, prob_proxy_prompt, prob_plain_text_prompt)
    else:
        raise ValueError(f"Unknown region type: {region_type}")


class DenseDetectionDatasetForQwen2_5_VL(Dataset):
    region_types = ['hbb', 'obb', 'polygon']  # TODO: maybe support seg for iSAID

    def __init__(
        self, 
        data_config_list: list[dict],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_for_position_ids: Optional[transformers.PreTrainedModel] = None,
    ):
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
        self.keep_empty_gt = getattr(self.data_args, 'keep_empty_gt', True)
        self.prob_open_end_prompts = getattr(self.data_args, 'prob_open_end_prompts', 0.0)
        list_data_dict = []
        for i, data in enumerate(data_config_list):
            metadata = load_json_or_jsonl(data["annotation_path"])
            annotations = preprocess_coco_format_metadata(metadata, self.keep_empty_gt)
            annotations = random_sample(data, annotations, getattr(data_args, 'data_sampling_seed', None))
            logging.info(f"sampling {len(annotations)} examples from {data_config_list[i]}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
                ann["image_file_path"] = os.path.join(ann["data_path"], ann['file_name'])
                ann["use_synonyms"] = True if random.random() <= self.prob_open_end_prompts else False
                ann["synonyms_dict"] = data.get("synonyms_dict", None)
            list_data_dict += annotations        
        self.list_data_dict = list_data_dict

        todo_list = [
            (metainfo_id, region_type)
            for region_type in self.region_types if region_type != 'seg'
            for metainfo_id in range(len(self.list_data_dict))
        ]
        self.todo_list = todo_list

    def __len__(self):
        return len(self.todo_list)
        
    def add_resized_size_to_metainfo(self, meta):
        original_width, original_height = meta["image_width"], meta["image_height"]

        resized_height, resized_width = smart_resize(
            original_height, original_width, factor=self.image_factor,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        
        meta["resized_width"] = resized_width
        meta["resized_height"] = resized_height
        return meta

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
            meta = self.random_resize(meta)
        else:
            self.add_resized_size_to_metainfo(meta)
        return get_messages(region_type, meta, self.prob_proxy_prompt, self.prob_plain_text_prompt)

    def preprocess(self, messages, **kwargs):
        return process_to_train_qwen2_5_vl_with_default_chat_template(
            self.processor, messages, 
            min_pixels=self.min_pixels, 
            max_pixels=self.max_pixels,
            model_for_position_ids=self.model_for_position_ids,
        )
                
    def __getitem__(self, idx):
        messages = self.get_messages(idx)
        return self.preprocess(messages)


class DenseDetectionHbbOnlyDatasetForQwen2_5_VL(DenseDetectionDatasetForQwen2_5_VL):
    region_types = ['hbb']


class DenseDetectionPolyOnlyDatasetForQwen2_5_VL(DenseDetectionDatasetForQwen2_5_VL):
    region_types = ['polygon']
