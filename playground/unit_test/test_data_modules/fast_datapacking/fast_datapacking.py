import sys
sys.path.insert(0, "/mnt/petrelfs/liqingyun/msr/code/rscovlm")

import base64
import re
import requests
import binpacking
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from typing import Optional

from PIL import Image
from io import BytesIO

from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

from rscovlm.training.data import make_supervised_data_module
from qwen_vl_utils import smart_resize, process_vision_info
from qwen_vl_utils.vision_process import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS

def get_length_qwen2_5_vl_with_default_chat_template(
        processor: Qwen2_5_VLProcessor, 
        messages: list[dict], 
        add_vision_id: bool = False, 
        add_generation_prompt: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        model_for_position_ids: Optional[Qwen2_5_VLForConditionalGeneration] = None,
        return_tensors: str = 'pt',
        prob_random_resize=1.,
    ) -> int:

    image_processor = processor.image_processor
    merge_length = image_processor.merge_size ** 2
    temporal_patch_size = image_processor.temporal_patch_size
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size

    output = []  # (text, is_label)
    image_count = 0

    # this section is built from the jinja code for Qwen2.5-VL default chat template
    for i, message in enumerate(messages):
        if i == 0 and message['role'] != 'system':
            output.append(["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", False])
        output.append([f"<|im_start|>{message['role']}\n", False])
        if isinstance(message['content'], str):
            output.append([message['content'], message['role'] == 'assistant'])
            output.append(["<|im_end|>", message['role'] == 'assistant'])
            output.append(["\n", False])
        else:
            for content in message['content']:
                if content['type'] == 'image' or 'image' in content or 'image_url' in content: # each image
                    assert message['content'] != 'assistant', message['content']

                    # maybe add vision id
                    image_count += 1
                    if add_vision_id:
                        output.append([f"Picture {image_count}: ", False])

                    # deal with image min_pixels/max_pixels
                    if min_pixels is None: 
                        assert image_processor.min_pixels == image_processor.size["shortest_edge"]
                        min_pixels = image_processor.min_pixels
                    content['min_pixels'] = min_pixels
                    if max_pixels is None:
                        assert image_processor.max_pixels == image_processor.size["longest_edge"]
                        max_pixels = image_processor.max_pixels
                    content['max_pixels'] = max_pixels
                    
                    # fetch image resize w/o decode pixels
                    H, W = fast_fetch_image(content)

                    # get processed image size
                    patches_shape = [temporal_patch_size, 3, H, W]  # [N, C, H, W]
                    grid_t = patches_shape[0] // temporal_patch_size
                    grid_h, grid_w = H // patch_size, W // patch_size
                    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])

                    # get text inputs
                    assert image_grid_thw.prod() % merge_length == 0, (image_grid_thw, merge_length)
                    num_tokens = image_grid_thw.prod() // merge_length
                    output.append([f"<|vision_start|>{'<|image_pad|>'*num_tokens}<|vision_end|>", False])
                    
                elif 'text' in content:
                    output.append([content['text'], message['role'] == 'assistant'])

            output.append(["<|im_end|>", message['role'] == 'assistant'])
            output.append(["\n", False])
    if add_generation_prompt:
        output.append(["<|im_start|>assistant\n", False])
    # post_process and tokenizer
    all_input_ids_ = []
    simplified_output = deepcopy(output[0])
    len_input_ids_ = 0
    for text, is_label in output[1:]:
        if is_label == simplified_output[1]: # merge output with the same label
            simplified_output[0] = "".join([simplified_output[0], text]) # 
        else:
            input_ids_ = processor.tokenizer(simplified_output[0], add_special_tokens=False, padding=False, return_tensors=return_tensors)['input_ids'][0]
            all_input_ids_.append(input_ids_)
            simplified_output = [text, is_label]
    # the last message
    input_ids_ = processor.tokenizer(simplified_output[0], add_special_tokens=False, padding=False, return_tensors=return_tensors)['input_ids'][0]
    all_input_ids_.append(input_ids_)
    if isinstance(all_input_ids_[0], torch.Tensor):
        input_ids_ = torch.cat(all_input_ids_, dim=0).to(torch.long)
    else:
        input_ids_ = np.concatenate(all_input_ids_, axis=0).astype(np.int64)
    len_input_ids_ = len(input_ids_)
    # simpler method:
    # image_meta = []
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_pad_numbers = []
    for msg in messages:
        if isinstance(msg['content'], list):
            for content in msg['content']:
                if isinstance(content, dict) and content.get("type") == "image": # image

                    # fetch image resize w/o decode pixels
                    H, W = fast_fetch_image(content)

                    # get processed image size
                    patches_shape = [temporal_patch_size, 3, H, W]  # [N, C, H, W]
                    grid_t = patches_shape[0] // temporal_patch_size
                    grid_h, grid_w = H // patch_size, W // patch_size
                    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])

                    # get text inputs
                    assert image_grid_thw.prod() % merge_length == 0, (image_grid_thw, merge_length)
                    num_tokens = image_grid_thw.prod() // merge_length
                    image_pad_numbers.append(num_tokens)
    img_idx = 0
    def replacer(_match):
        nonlocal img_idx
        num_tokens = image_pad_numbers[img_idx]
        img_idx += 1
        return "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"
    pattern = r"<\|vision_start\|><\|image_pad\|><\|vision_end\|>"
    rendered_with_vision_message = re.sub(pattern, replacer, rendered)
    input_ids = processor.tokenizer(rendered_with_vision_message, add_special_tokens=False, padding=False, return_tensors=return_tensors)['input_ids'][0]
    assert input_ids_.equal(input_ids)
    return len_input_ids_

        
def fast_fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "resized_height" in ele and "resized_width" in ele: # no need of image itself
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        # print(ele)
        if "image" in ele:
            image = ele["image"]
        else:
            image = ele["image_url"]
        image_obj = None
        if isinstance(image, Image.Image):
            image_obj = image
        elif image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image, stream=True)
            image_obj = Image.open(BytesIO(response.content))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])

        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            image_obj = Image.open(image)
        if image_obj is None:
            raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
        width, height = image_obj.size
        # width, height = image_obj.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return resized_height, resized_width

def data_generator(datasets, model_path, max_length, packing_interval):

    data_args = Namespace(
    datasets=datasets,
    flatten_data=True,
    data_path=None,
    image_folder=None,
    min_pixels=256 * 28 * 28,  # 4 * 28 * 28
    max_pixels=1296 * 28 * 28,  # 16384 * 28 * 28
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    modules = make_supervised_data_module(processor, data_args, model)
    train_dataset, data_collator = modules["train_dataset"], modules["data_collator"]

    data_to_packing = []
    for i in tqdm(range(len(train_dataset)), desc='Process and get_length:'):
        messages = train_dataset.get_messages(i)
        len_input_ids = get_length_qwen2_5_vl_with_default_chat_template(
            processor, messages, 
            min_pixels=data_args.min_pixels, 
            max_pixels=data_args.max_pixels,
            model_for_position_ids=model,
            )

        data_to_packing.append((messages, len_input_ids))
        
        if len(data_to_packing) >= packing_interval:
            packed_data = binpacking.to_constant_volume(data_to_packing,  max_length, weight_pos=1)

            yield from packed_data
            data_to_packing.clear()
    # 处理剩下未打包的数据
    if data_to_packing:
        packed_data = binpacking.to_constant_volume(data_to_packing, max_length, weight_pos=1)
        yield from packed_data
    return


if __name__ == "__main__":
    dataset_names = ["vhm_dataset"] # ["dota_trainval512"] # all resized, [vhm_dataset] # no resized
    max_length = 4096
    packing_interval=32
    model_path = "./playground/Qwen2.5-VL-3B-Instruct"
    for packed_bin in data_generator(dataset_names, model_path, max_length, packing_interval):
        # print(packed_bin)
        continue

