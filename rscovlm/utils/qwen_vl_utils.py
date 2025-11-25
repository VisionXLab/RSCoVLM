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
