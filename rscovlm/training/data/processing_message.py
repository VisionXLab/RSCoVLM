import inspect
from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature

from qwen_vl_utils import fetch_video, fetch_image, smart_resize
from qwen_vl_utils.vision_process import VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS

def get_qwenvl_rope_index(model, **kwargs):
    is_batch_data = True
    if 'input_ids' in kwargs and kwargs['input_ids'] is not None and len(kwargs['input_ids'].shape) == 1:
        kwargs['input_ids'] = kwargs['input_ids'].unsqueeze(0)
        is_batch_data = False
    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None and len(kwargs['attention_mask'].shape) == 1:
        kwargs['attention_mask'] = kwargs['attention_mask'].unsqueeze(0)
    signature = inspect.signature(model.get_rope_index)
    allowed_keys = signature.parameters.keys()
    kwargs = {key: kwargs[key] for key in kwargs if key in allowed_keys}
    rope_index, _ = model.get_rope_index(**kwargs)
    return rope_index if is_batch_data else rope_index.squeeze(1)


def process_to_train_qwen2_5_vl_with_default_chat_template(
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
    ) -> BatchFeature:
    """
    Process the messages of a training sample to the model inputs for training Qwen2.5-VL models.

    The API is easy: requires a `Qwen2_5_VLProcessor` and a qwen-style openai-like chat `messages`, 
    and outputs `BatchFeature` (as `processor.__call__()`) as the model inputs.

    This function is customized as a replacement of `Qwen2_5_VLProcessor.__call__()` and `qwen_vl_utils.process_vision_info()`. 
    The official processing suffers from: (official: firstly extract vision info with `qwen_vl_utils`, then preprocess vision and text with `Qwen2_5_VLProcessor`)
    1. It seems mainly for inference, not training. The jinja template is not convinient for get the `labels` aside `input_ids`, especially when we only want a section of the text as labels.
    2. The `process_vision_info` and `Qwen2_5_VLProcessor` process the vision section for two times.
    In this function, we customize a pipe to simplify the process.

    NOTE: The messages interpreting refers to the default jinja template of `Qwen2.5-VL` series.
    You should check and modify the code if the `chat_template` is changed.
    """
    image_processor = processor.image_processor
    merge_length = image_processor.merge_size ** 2

    output = []  # (text, is_label)
    all_pixel_values = []
    all_image_grid_thw = []
    all_video_pixel_values = []
    all_video_grid_thw = []
    all_second_per_grid_ts = []
    image_count, video_count = 0, 0

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
                if content['type'] == 'image' or 'image' in content or 'image_url' in content:
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

                    # get image inputs
                    image_input = image_processor(images=fetch_image(content), videos=None, return_tensors=return_tensors, min_pixels=min_pixels, max_pixels=max_pixels)
                    pixel_values = image_input["pixel_values"]
                    image_grid_thw = image_input["image_grid_thw"]
                    all_pixel_values.append(pixel_values)
                    all_image_grid_thw.append(image_grid_thw)

                    # get text inputs
                    assert image_grid_thw.prod() % merge_length == 0, (image_grid_thw, merge_length)
                    num_tokens = image_grid_thw.prod() // merge_length
                    output.append([f"<|vision_start|>{'<|image_pad|>'*num_tokens}<|vision_end|>", False])

                elif content['type'] == 'video' or 'video' in content:
                    assert message['content'] != 'assistant', message['content']

                    # maybe add vision id
                    video_count += 1
                    if add_vision_id:
                        output.append([f"Video {video_count}: ", False])

                    # deal with video_min_pixels/video_max_pixels
                    if video_min_pixels is None:
                        video_min_pixels = VIDEO_MIN_PIXELS
                    content['min_pixels'] = video_min_pixels
                    if video_max_pixels is None:
                        video_max_pixels = VIDEO_MAX_PIXELS
                    content['max_pixels'] = video_max_pixels

                    # get video inputs
                    video_input, video_sample_fps = fetch_video(content, return_video_sample_fps=True)
                    video_input = image_processor(images=None, videos=video_input, return_tensors=return_tensors, min_pixels=video_min_pixels, max_pixels=video_max_pixels)
                    pixel_values_videos = video_input["pixel_values_videos"]
                    video_grid_thw = video_input["video_grid_thw"]
                    second_per_grid_ts = image_processor.temporal_patch_size / video_sample_fps
                    all_video_pixel_values.append(pixel_values_videos)
                    all_video_grid_thw.append(video_grid_thw)
                    all_second_per_grid_ts.append(second_per_grid_ts)

                    # get text inputs
                    assert video_grid_thw.prod() % merge_length == 0, (video_grid_thw, merge_length)
                    num_tokens = video_grid_thw.prod() // merge_length
                    output.append([f"<|vision_start|>{'<|video_pad|>'*num_tokens}<|vision_end|>", False])
                    
                elif 'text' in content:
                    output.append([content['text'], message['role'] == 'assistant'])

            output.append(["<|im_end|>", message['role'] == 'assistant'])
            output.append(["\n", False])

    if add_generation_prompt:
        output.append(["<|im_start|>assistant\n", False])

    # concatenate vision inputs
    vision_inputs = {}
    if len(all_pixel_values) > 0:
        if isinstance(all_pixel_values[0], torch.Tensor):
            vision_inputs["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            vision_inputs["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        else:  # actually useless
            vision_inputs["pixel_values"] = np.concatenate(all_pixel_values, axis=0)
            vision_inputs["image_grid_thw"] = np.concatenate(all_image_grid_thw, axis=0)
    if len(all_video_pixel_values) > 0:
        if isinstance(all_video_pixel_values[0], torch.Tensor):
            vision_inputs["pixel_values_videos"] = torch.cat(all_video_pixel_values, dim=0)
            vision_inputs["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
            vision_inputs["second_per_grid_ts"] = torch.tensor(all_second_per_grid_ts).to(torch.float32)
        else:  # actually useless
            vision_inputs["pixel_values_videos"] = np.concatenate(all_video_pixel_values, axis=0)
            vision_inputs["video_grid_thw"] = np.concatenate(all_video_grid_thw, axis=0)
            vision_inputs["second_per_grid_ts"] = all_second_per_grid_ts
    
    # concatenate text sections (optional)
    simplified_output = [deepcopy(output[0])]
    for text, is_label in output[1:]:
        if is_label == simplified_output[-1][1]:
            simplified_output[-1][0] = "".join([simplified_output[-1][0], text])
        else:
            simplified_output.append([text, is_label])
    output = simplified_output

    # tokenize text inputs and get labels for training
    all_input_ids, all_labels = [], []
    for text, is_label in output:
        input_ids = processor.tokenizer(text, add_special_tokens=False, padding=False, return_tensors=return_tensors)['input_ids'][0]
        if isinstance(input_ids, torch.Tensor):
            labels = input_ids if is_label else torch.full_like(input_ids, -100)
        else:
            labels = input_ids if is_label else np.full_like(input_ids, -100)

        all_input_ids.append(input_ids)
        all_labels.append(labels)
    if isinstance(all_input_ids[0], torch.Tensor):
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)  # TODO: check whether it is nessary
    else:
        input_ids = np.concatenate(all_input_ids, axis=0).astype(np.int64)
        labels = np.concatenate(all_labels, axis=0).astype(np.int64)
        attention_mask = (input_ids > -1000000).astype(np.int64)

    text_inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    if model_for_position_ids is not None:
        text_inputs['position_ids'] = get_qwenvl_rope_index(model_for_position_ids, **text_inputs, **vision_inputs)
    return BatchFeature(data={**text_inputs, **vision_inputs})
