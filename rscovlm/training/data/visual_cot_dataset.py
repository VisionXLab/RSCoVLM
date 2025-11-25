import os
import logging
from PIL import Image
from copy import deepcopy
from typing import Optional

from torch.utils.data import Dataset

import transformers

from qwen_vl_utils import smart_resize
from ..params import DataArguments
from .processing_message import process_to_train_qwen2_5_vl_with_default_chat_template
from rscovlm.utils import load_json_or_jsonl, random_sample

Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size to avoid errors with large images


SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "image_zoom_in", "description": "Zoom in to the region of interest on the given image. Input the bounding box of the region of interest and return the zoomed-in image to help to focus and view more clearly.", "parameters": {"name": "bbox", "type": "list[int]", "description": "The bounding box of the region of interest as an array [x1, y1, x2, y2], where (x1, y1) and (x2, y2) indicate the positions' absolute coordinates of the left top and right down corners, respectively.", "required": true}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


TOOL_CALL_TEMPLATE = """<tool_call>\n{{"name": "image_zoom_in", "arguments": {{"bbox": [{x1}, {y1}, {x2}, {y2}]}}}}\n</tool_call>"""


def get_messages(meta, disable_visual_cot=False):
    bbox_info = meta['bbox_info']
    image_file_path = meta["image_file_path"]

    if 'question' in meta and 'answer' in meta:
        question = meta['question']
        answer = meta['answer']
    elif 'Text' in meta and 'Ground truth' in meta and 'Answer choices' in meta:
        choice_prompt = ' The choices are listed below: \n'
        for choice in meta['Answer choices']:
            choice_prompt += choice + "\n"
        question = (
            meta['Text']
            .rstrip('\n Answer the question using a single word or phrase.') 
            + choice_prompt 
            + 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.' 
            + '\nThe best answer is:'
        )
        answer = meta['Ground truth']
    else:
        raise ValueError("Meta information must contain either 'question' and 'answer' or 'Text', 'Ground truth', and 'Answer choices'.")

    pil_full_image = Image.open(image_file_path).convert("RGB")
    image_content = {"type": "image", "image": pil_full_image}
    if 'resized_width' in meta or 'resized_height' in meta:
        image_content["resized_width"] = meta['resized_width']
        image_content["resized_height"] = meta['resized_height']

    crop_hbox = bbox_info['crop_hbox']

    # In this version, we use `crop_hbox` to crop the region, and do not adopt visual cot on examples with multiple `crop_hbox` (e.g., counting, compare tasks)
    if len(crop_hbox) == 1 and not disable_visual_cot:
        pil_cropped_image = pil_full_image.crop(crop_hbox[0])
        cropped_image_content = {"type": "image", "image": pil_cropped_image}

        if 'resized_width' in meta or 'resized_height' in meta:
            for bbox in bbox_info.get('rbox', []) + bbox_info['crop_hbox']:
                for i in range(0, len(bbox), 2):
                    bbox[i] = int(bbox[i] * meta['resized_width'] / meta['image_ori_width'])
                    bbox[i + 1] = int(bbox[i + 1] * meta['resized_height'] / meta['image_ori_height'])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": f"{question}\nZoom in to the region of interest and answer the question."}  # TODO: do we require `answer the question using a single word or pharse`
            ]},
            {"role": "assistant", "content": TOOL_CALL_TEMPLATE.format(
                x1=int(crop_hbox[0][0]), y1=int(crop_hbox[0][1]), x2=int(crop_hbox[0][2]), y2=int(crop_hbox[0][3]))},
            {"role": "user", "content": [
                cropped_image_content,
                {"type": "text", "text": "Here's the zoomed-in image. You can complete your task now."}
            ]},
            {"role": "assistant", "content": answer}
        ]
    else:
        messages = [
            {"role": "user", "content": [
                image_content,
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
    return messages


class VisualCOTDatasetForQwen2_5_VL(Dataset):

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
        self.disable_visual_cot = getattr(self.data_args, "disable_visual_cot", False)

        for meta in self.list_data_dict:
            meta["image"] = meta["image"].replace('/gruntdata/rs_nas/workspace/junhong.ljw/dataset/', '')  # TODO: delete the private path before release
            meta["image_file_path"] = os.path.join(meta["data_path"], meta['image'])

    def __len__(self):
        return len(self.list_data_dict)
        
    def add_resized_size_to_metainfo(self, meta):
        original_width, original_height = meta["image_ori_width"], meta["image_ori_height"]

        resized_height, resized_width = smart_resize(
            original_height, original_width, factor=self.image_factor,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        
        meta["resized_width"] = resized_width
        meta["resized_height"] = resized_height
        return meta

    def get_messages(self, idx):
        meta = deepcopy(self.list_data_dict[idx])
        self.add_resized_size_to_metainfo(meta)
        return get_messages(meta, self.disable_visual_cot)

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
