import os
import random
import logging
from PIL import Image
from typing import Optional

import transformers
from datasets import load_from_disk, Dataset

from qwen_vl_utils.vision_process import fetch_image, smart_resize

from ..params import DataArguments
from ...utils import load_json_or_jsonl
from .sft_dataset import get_messages_from_llava_style_conversations, SupervisedDatasetForQwen2_5_VL, random_resize_images_in_messages


def random_sample(data, ds: Dataset, seed: Optional[int] = None):
    if seed is not None:
        seed = 42
    rnd = random.Random(seed)    

    sampling_rate = data.get("sampling_rate", 1.0)
    total_samples = len(ds)
    if sampling_rate < 1.0:
        sample_num = max(1, int(total_samples * sampling_rate))
        indices = rnd.sample(range(total_samples), sample_num)
    elif sampling_rate > 1.0:
        assert sampling_rate.is_integer()
        sampling_rate = int(sampling_rate)
        indices = list(range(total_samples)) * sampling_rate
    else:
        return ds
    return ds.select(indices)


class LlavaOneVisionDatasetForQwen2_5_VL(SupervisedDatasetForQwen2_5_VL):
    """Processed from https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data"""

    def __init__(
        self, 
        data_config_list: list[dict],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_for_position_ids: Optional[transformers.PreTrainedModel] = None,
    ):
        # Get data list containing the full path
        list_data_meta = None
        for i, data in enumerate(data_config_list):
            ds = load_from_disk(data["data_path"])
            ds = random_sample(data, ds, seed=getattr(data_args, 'data_sampling_seed', None))
            logging.info(f"sampling {len(ds)} examples from {data_config_list[i]}")
            if list_data_meta is None:
                list_data_meta = ds
            else:
                list_data_meta += ds
        
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

    def conversations_has_image(self, conversations):
        for conv in conversations:
            if "<image>" in conv["value"]:
                return True
        return False

    def ensure_image_tag_in_data_meta(self, data_meta):
        if data_meta["image"] is None:
            return data_meta
        
        conversations = data_meta["conversations"]
        if not self.conversations_has_image(conversations):
            first_human_conv = next(conv for conv in conversations if conv["from"] == "human")
            first_human_conv["value"] = "<image>\n" + first_human_conv['value']

    def get_messages(self, idx):
        data_meta = self.list_data_meta[idx]  # a single data item
        self.ensure_image_tag_in_data_meta(data_meta)

        if isinstance(data_meta, dict) and 'conversations' in data_meta and isinstance(data_meta['conversations'], list):
            messages = get_messages_from_llava_style_conversations(data_meta)
        elif isinstance(data_meta, list):
            messages = data_meta
        else:
            raise ValueError(f"Invalid data format: {data_meta}")

        if random.random() <= self.prob_random_resize:
            messages = random_resize_images_in_messages(messages, min_pixels=self.min_pixels, max_pixels=self.max_pixels, image_factor=self.image_factor)
        return messages

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            # load next item
            print(f"Error in __getitem__ for idx {idx}: {e}")
            return self.__getitem__(idx + 1)
