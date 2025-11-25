import os
import re
import magic
import random
import logging
import datetime
from PIL import Image
from copy import deepcopy
from typing import Optional

from torch.utils.data import Dataset
import transformers

from qwen_vl_utils.vision_process import fetch_image, smart_resize

from ..params import DataArguments
from .processing_message import process_to_train_qwen2_5_vl_with_default_chat_template
from rscovlm.utils import load_json_or_jsonl, random_sample  # TODO: add param to control all randomness

def add_image_folder_to_images(data, image_folder):
    for item in data:
        if item["role"] == "user":
            for content in item["content"]:
                if content.get("type") == "image":
                    path = content["image"]
                    prefix = "file://"
                    if path.startswith(prefix):
                        original_path = path[len(prefix):]
                        if original_path.startswith("/"):
                            new_path = f"{prefix}{image_folder}{original_path}"
                        else:
                            new_path = f"{prefix}{image_folder}/{original_path}"
                        content["image"] = new_path
    return data

def is_video_file(path):
    if isinstance(path, str):
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')):
            return True
        try:
            mime_type = magic.from_file(path, mime=True)
            return mime_type.startswith('video/')
        except:
            return False
    else:
        return False


def is_image_file(path):
    if isinstance(path, str):
        if path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tif', '.tiff')):
            return True
        try:
            mime_type = magic.from_file(path, mime=True)
            return mime_type.startswith('image/')
        except:
            return False
    else:
        return False


def is_image_or_image_file(item):
    if isinstance(item, str):
        return is_image_file(item)
    elif isinstance(item, Image.Image):
        return True
    else:
        return False


def get_images_from_meta(meta):
    # support single image and multiple images
    meta_images = deepcopy(meta.get("image", []))  # Union[str, list[str]]
    if not isinstance(meta_images, list):
        meta_images = [meta_images]
    if not all(is_image_or_image_file(img) for img in meta_images):
        raise ValueError(f"Invalid image format: {meta_images}")
    return [os.path.join(meta["data_path"], img) if isinstance(img, str) else img for img in meta_images]


def get_videos_from_meta(meta):
    # support single video and multiple videos, each video support video file (to be decoded) or video frames (pre-decoded)
    meta_videos = deepcopy(meta.get("video", []))  # str (single video file), list[str] (frames of single video or multiple video files), list[list[str]] (frames of multiple videos)
    
    if isinstance(meta_videos, str) and is_video_file(meta_videos):  # single video file
        return [os.path.join(meta["data_path"], meta_videos)]
    elif isinstance(meta_videos, list) and all(is_video_file(item) for item in meta_videos):  # multiple video files
        return [os.path.join(meta["data_path"], video) for video in meta_videos]
    elif isinstance(meta_videos, list) and all(is_image_file(item) for item in meta_videos):  # frames of single video

        if 'timestamps' in meta:
            # found in TEOChatlas instruct.json
            if len(meta['timestamps']) not in [len(meta_videos), 0]:
                raise ValueError(f"Number of frames ({len(meta_videos)}) does not match number of timestamps ({len(meta['timestamps'])})")
            # sort frames by timestamps
            if len(meta['timestamps']) > 0:
                try:
                    meta_videos, meta_timestamps = zip(*sorted(
                        zip(meta_videos, meta['timestamps']),
                        key=lambda t: datetime.strptime(t[1], "%Y-%m-%d")
                    ))
                except ValueError as e:
                    raise NotImplementedError(f"Not implemented timestamp format: {meta['timestamps']}") from e
        
        return [[os.path.join(meta["data_path"], frame) for frame in meta_videos]]
    elif isinstance(meta_videos, list) and all(isinstance(item, list) for item in meta_videos) \
        and all(all(is_video_file(frame) for frame in video) for video in meta_videos):  # frames of multiple videos
        return [[os.path.join(meta["data_path"], frame) for frame in video] for video in meta_videos]
    else:
        raise ValueError(f"Invalid video format: {meta_videos}")


def get_image_content(img, meta):
    if isinstance(img, str):
        img_path = img
        image_content = f"file://{img_path}"
    elif isinstance(img, Image.Image):
        image_content = img
    else:
        raise ValueError(f"Invalid image: {img}")
    
    if 'resized_width' in meta or 'resized_height' in meta:
        image_content = {
            "type": "image",
            "image": image_content,
            "width": meta["resized_width"],
            "height": meta["resized_height"],
        }
    else:
        image_content = {"type": "image", "image": image_content}
    return image_content


def get_video_content(video, meta):
    if isinstance(video, str) and is_video_file(video):
        video_content = f"file://{video}"
    elif isinstance(video, list) and all(is_image_file(frame) for frame in video):
        video_content = [f"file://{frame}" for frame in video]
    elif isinstance(video, list) and all(isinstance(frame, Image.Image) for frame in video):
        video_content = video
    else:
        raise ValueError(f"Invalid video: {video}")

    if 'resized_width' in meta or 'resized_height' in meta:
        video_content = {
            "type": "video",
            "video": video_content,
            "width": meta["resized_width"],
            "height": meta["resized_height"],
        }
    else:
        video_content = {"type": "video", "video": video_content}
    return video_content


def get_messages_from_llava_style_conversations(meta):
    images = get_images_from_meta(meta)
    videos = get_videos_from_meta(meta)
    
    messages = []
    for turn in meta["conversations"]:
        try:
            role = {"human": "user", "gpt": "assistant"}[turn["from"]]
        except:
            import ipdb; ipdb.set_trace()
            raise ValueError(f"Invalid role: {turn}")

        value = turn["value"]
        # WARNING: There's a hardcode to strip the '\n' behind the <image> and <video> tokens
        value = value.replace("<image>\n", "<image>").replace("<video>\n", "<video>").strip()

        content = []
        for part in re.split(r'(<image>|<video>)', value):
            if part.strip() == "":
                continue
            if part == "<image>":
                content.append(get_image_content(images.pop(0), meta))
            elif part == "<video>":
                content.append(get_video_content(videos.pop(0), meta))
            else:
                content.append({"type": "text", "text": part})
        msg = {"role": role, "content": content}
        messages.append(msg)

    return messages


def random_resize_images_in_messages(messages, min_pixels, max_pixels, image_factor):
    # NOTE: we only resize images, but do not resize videos
    for msg in messages:
        if msg['role'] == 'user' and isinstance(msg['content'], list):
            for content in msg['content']:
                if content['type'] == 'image':
                    image = fetch_image(content)
                    image_width, image_height = image.size
                    original_pixels = image_width * image_height
                    
                    scale = random.uniform(min_pixels / original_pixels, max_pixels / original_pixels)
                    resized_height, resized_width = int(image_height * scale), int(image_width * scale)
                    resized_height, resized_width = smart_resize(
                        resized_height, resized_width, factor=image_factor,
                        min_pixels=min_pixels, max_pixels=max_pixels)

                    image.resize((resized_width, resized_height), Image.BICUBIC)
                    content['image'] = image
    return messages


class SupervisedDatasetForQwen2_5_VL(Dataset):

    def __init__(
        self, 
        data_config_list: list[dict],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_for_position_ids: Optional[transformers.PreTrainedModel] = None,
    ):
        # Get data list containing the full path
        list_data_meta = []
        for i, data in enumerate(data_config_list):
            annotations = load_json_or_jsonl(data["annotation_path"])  # anno contents
            annotations = random_sample(data, annotations, getattr(data_args, 'data_sampling_seed', None))
            logging.info(f"sampling {len(annotations)} examples from {data_config_list[i]}")
            for ann in annotations:  # TODO: fixme
                ann["data_path"] = data["data_path"]
            list_data_meta += annotations
        
        self.processor = processor
        self.list_data_meta = list_data_meta
        self.data_args = data_args
        self.model_for_position_ids = model_for_position_ids
        
        self.min_pixels = getattr(self.data_args, 'min_pixels', processor.image_processor.min_pixels)
        self.max_pixels = getattr(self.data_args, 'max_pixels', processor.image_processor.max_pixels)
        self.video_min_pixels = getattr(self.data_args, 'video_min_pixels', processor.image_processor.min_pixels)
        self.video_max_pixels = getattr(self.data_args, 'video_max_pixels', processor.image_processor.max_pixels)
        self.patch_size = self.processor.image_processor.patch_size
        self.merge_size = self.processor.image_processor.merge_size
        self.image_factor = self.patch_size * self.merge_size
        self.prob_random_resize = getattr(self.data_args, 'prob_random_resize', 0)
        
    def __len__(self):
        return len(self.list_data_meta)

    def get_messages(self, idx):
        data_meta = self.list_data_meta[idx]  # a single data item
        if isinstance(data_meta, dict) and 'conversations' in data_meta and isinstance(data_meta['conversations'], list):
            messages = get_messages_from_llava_style_conversations(data_meta)
        elif isinstance(data_meta, list):
            messages = add_image_folder_to_images(data_meta, self.data_args.image_folder)    # TODO: check and fix
        else:
            raise ValueError(f"Invalid data format: {data_meta}")

        if random.random() <= self.prob_random_resize:
            messages = random_resize_images_in_messages(messages, min_pixels=self.min_pixels, max_pixels=self.max_pixels, image_factor=self.image_factor)
        return messages
    
    def preprocess(self, messages, **kwargs):
        return process_to_train_qwen2_5_vl_with_default_chat_template(
            self.processor, messages, 
            min_pixels=self.min_pixels, 
            max_pixels=self.max_pixels,
            video_min_pixels=self.video_min_pixels,
            video_max_pixels=self.video_max_pixels,
            model_for_position_ids=self.model_for_position_ids,
        )
                
    def __getitem__(self, idx):
        messages = self.get_messages(idx)
        return self.preprocess(messages)
