import copy
import os
from typing import Dict
import torch
import transformers
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re


import base64
import requests
from io import BytesIO
from typing import Optional

from PIL import Image
import torch

import qwen_vl_utils
from qwen_vl_utils.vision_process import (
    fetch_image, fetch_video, extract_vision_info, 
    smart_resize, to_rgb, IMAGE_FACTOR
)


Image.MAX_IMAGE_PIXELS = None
IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR, return_image_size: bool = False) -> Image.Image:
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
    
    original_width, original_height = image_obj.size

    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", qwen_vl_utils.vision_process.MIN_PIXELS)
        max_pixels = ele.get("max_pixels", qwen_vl_utils.vision_process.MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    if return_image_size:
        return image, original_width, original_height, resized_width, resized_height
    else:
        return image


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_image_sizes: bool = False,
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    image_size_list = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_input, original_width, original_height, resized_width, resized_height = fetch_image(vision_info, return_image_size=True)
            image_size_list.append((original_width, original_height, resized_width, resized_height))
            image_inputs.append(image_input)
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    args = []
    if return_image_sizes:
        args.append(image_size_list)
    if return_video_kwargs:
        args.append({'fps': video_sample_fps_list})
    return image_inputs, video_inputs, *args


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future  # TODO
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixels": min_pixel,
                "max_pixels": max_pixel

            }
            ]
        }
    ]

    image_input, _, image_sizes_list = process_vision_info(messages, return_image_sizes=True)
    # print(image_sizes_list)
    return image_input[0], image_sizes_list[0], messages  # TODO: single image?


def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future  # TODO

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs, messages


def apply_qwen2_5_vl_default_chat_template(messages, add_vision_id=False, add_generation_prompt=False):
    """
    {% set image_count = namespace(value=0) %}
    {% set video_count = namespace(value=0) %}
    {% for message in messages %}
        {% if loop.first and message['role'] != 'system' %}
            <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        {% endif %}
            <|im_start|>{{ message['role'] }}\n
        {% if message['content'] is string %}
            {{ message['content'] }}<|im_end|>\n
        {% else %}
            {% for content in message['content'] %}
                {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                    {% set image_count.value = image_count.value + 1 %}
                    {% if add_vision_id %}
                        Picture {{ image_count.value }}: 
                    {% endif %}
                    <|vision_start|><|image_pad|><|vision_end|>
                {% elif content['type'] == 'video' or 'video' in content %}
                    {% set video_count.value = video_count.value + 1 %}
                    {% if add_vision_id %}
                        Video {{ video_count.value }}: 
                    {% endif %}
                    <|vision_start|><|video_pad|><|vision_end|>
                {% elif 'text' in content %}
                    {{ content['text'] }}
                {% endif %}
            {% endfor %}
            <|im_end|>\n
        {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}
        <|im_start|>assistant\n
    {% endif %}
    """
    image_count, video_count = 0, 0
    output = []
    for i, message in enumerate(messages):
        if i == 0 and message['role'] != 'system':
            output.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        output.append(f"<|im_start|>{message['role']}\n")
        if isinstance(message['content'], str):
            output.append(f"{message['content']}<|im_end|>\n")
        else:
            for content in message['content']:
                if content['type'] == 'image' or 'image' in content or 'image_url' in content:
                    image_count += 1
                    if add_vision_id:
                        output.append(f"Picture {image_count}: ")
                    output.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif content['type'] == 'video' or 'video' in content:
                    video_count += 1
                    if add_vision_id:
                        output.append(f"Video {video_count}: ")
                    output.append("<|vision_start|><|video_pad|><|vision_end|>")
                elif 'text' in content:
                    output.append(content['text'])
            output.append("<|im_end|>\n")
    if add_generation_prompt:
        output.append("<|im_start|>assistant\n")
    return "".join(output)


def simplify_output(text):
    if isinstance(text, list):
        return [simplify_output(t) for t in text]
    patterns = [r'<\|image_pad\|>', r'<\|endoftext\|>', r'!']
    for pattern in patterns:
        text = re.sub(rf'({pattern})\1+', lambda m: f"{m.group(1)} * {len(m.group(0)) // len(m.group(1))} ", text)
    return text


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data: str | list,
        image_folder: str,  # TODO: make better
        processor: transformers.ProcessorMixin,
        data_args,
        model_id,  # TODO: remove this
        padding=True,  # TODO: refine tokenizer args
    ):
        if isinstance(data, str):
            self.list_data_dict = json.load(open(data, "r"))
        else:
            self.list_data_dict = data
        self.processor = processor
        self.data_args = data_args
        self.model_id = model_id
        self.padding = padding
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            image_sizes_list = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):  # TODO: consider this, and support aws s3
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                image_input, image_sizes, image_messages = get_image_info(image_file, self.data_args.min_pixels, self.data_args.max_pixels)
                images.append(image_input)
                image_sizes_list.append(image_sizes)

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs, video_messages = get_video_info(video_file, self.data_args.video_min_pixels, self.data_args.video_max_pixels, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this. # TODO: why not using apply_chat_template
        if sources[0]['role'] == 'system':
            system_message = sources.pop(0)['content']
        else:
            system_message = SYSTEM_MESSAGE
        
        if len(system_message) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{system_message}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)  # TODO: why > -1000000

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        # # DEBUG: check whether the hand-crafted input text is aligned with the that from apply_chat_template (here I only chat for image, ignore video)
        # if len(sources) == 2 and sources[0]['role'] == 'user' and sources[1]['role'] == 'assistant' and isinstance(sources[0]['content'], str) and len(image_messages) == 1:
        #     input_ids_decoded = processor.tokenizer.decode(input_ids)
        #     user_text_content = sources[0]['content']
        #     user_text_content = re.sub(r'<\|vision_start\|>(<\|image_pad\|>)+<\|vision_end\|>', '', user_text_content)
        #     user_image_round = image_messages[0]
        #     user_image_round['content'].append({"type": "text", 'text': user_text_content})
        #     new_message = [user_image_round, sources[1]]
        #     text = processor.apply_chat_template(new_message, tokenize=False)
        #     input_ids_2 = processor(text=[text], images=images, videos=videos, padding=False, return_tensors='pt')['input_ids'][0]
        #     input_ids_decoded_2 = processor.tokenizer.decode(input_ids_2)
        #     print(f"CHECKING: {input_ids_decoded == input_ids_decoded_2}")

        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_values_videos = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_values_videos.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            if isinstance(batch_pixel_values[0], torch.Tensor):
                data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
                data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)
            else:
                data_dict["pixel_values"] = np.concatenate(batch_pixel_values, axis=0)
                data_dict["image_grid_thw"] = np.concatenate(batch_image_thw, axis=0)

        if len(batch_pixel_values_videos) > 0:
            if isinstance(batch_pixel_values_videos[0], torch.Tensor):
                data_dict["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
                data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)
            else:
                data_dict["pixel_values_videos"] = np.concatenate(batch_pixel_values_videos, axis=0)
                data_dict["video_grid_thw"] = np.concatenate(batch_video_thw, axis=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    from rscovlm.training.data.refgeo_dataset import GeoGroundDatasetForQwen2_5_VL, GeoGroundSegOnlyDatasetForQwen2_5_VL, GeoGroundWoSegDatasetForQwen2_5_VL
    def make_single_train_dataset(dataset_type, data_path, image_folder):
        if dataset_type == 'geoground':
            DATASET_CLASS = GeoGroundDatasetForQwen2_5_VL
        elif dataset_type == 'geoground_seg_only':
            DATASET_CLASS = GeoGroundSegOnlyDatasetForQwen2_5_VL
        elif dataset_type == 'geoground_wo_seg':
            DATASET_CLASS = GeoGroundWoSegDatasetForQwen2_5_VL
        else:
            DATASET_CLASS = SupervisedDataset
        
        sft_dataset = DATASET_CLASS(
            data=data_path, image_folder=image_folder, 
            processor=processor, data_args=data_args, model_id=model_id
        )
        return sft_dataset
    
    if isinstance(data_args.data_path, list) and isinstance(data_args.dataset_type, list) and isinstance(data_args.image_folder, list):
        assert len(data_args.data_path) == len(data_args.dataset_type) == len(data_args.image_folder), \
            "data_path, data_args.image_folder and dataset_type must have the same length."
        sft_dataset = []
        for data_path, dataset_type, image_folder in zip(data_args.data_path, data_args.dataset_type, data_args.image_folder):
            sft_dataset.append(make_single_train_dataset(dataset_type, data_path, image_folder))
        sft_dataset = torch.utils.data.ConcatDataset(sft_dataset)
    else:
        sft_dataset = make_single_train_dataset(data_args.dataset_type, data_args.data_path, data_args.image_folder)

    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
