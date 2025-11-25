# We follow the [implementations of ms-swift](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/utils.py)
import re
import base64
import json
import random
import requests
import binpacking
import multiprocessing as mp
from io import BytesIO
from typing import Optional, Union, Tuple

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor

from qwen_vl_utils import smart_resize
from qwen_vl_utils.vision_process import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS

from .processing_message import process_to_train_qwen2_5_vl_with_default_chat_template

def shard_dataset(
    dataset: Union[Dataset, list[any]], 
    n_shards: int = 1, 
    shard_id: int = -1, 
    shuffle_seed: Optional[int] = None
) -> Union[Subset, list[Subset], list[any], list[list[any]]]:
    assert 0 < n_shards <= len(dataset), f"n_shards={n_shards} should be greater than 0 and less than or equal to the dataset size {len(dataset)}"
    assert hasattr(dataset, '__len__'), "Dataset must have a __len__ method"
    assert int(shard_id) < n_shards, f"shard_id={shard_id} should be less than n_shards={n_shards}"
    
    indices = list(range(len(dataset)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(indices)
    shard_size, remainder = divmod(len(indices), n_shards)

    if shard_id < 0:
        shards = []
        start = 0
        for i in range(n_shards):
            end = start + shard_size + (1 if i < remainder else 0)
            if isinstance(dataset, Dataset):
                shards.append(Subset(dataset, indices[start:end]))
            else:
                shards.append([dataset[i] for i in indices[start:end]])
            start = end
        return shards
    else:
        start = shard_id * shard_size + min(shard_id, remainder)
        end = start + shard_size + (1 if shard_id < remainder else 0)
        if isinstance(dataset, Dataset):
            return Subset(dataset, indices[start:end])
        else:
            return [dataset[i] for i in indices[start:end]]
            
def get_length_qwen2_5_vl_with_default_chat_template(
        processor: Qwen2_5_VLProcessor, 
        messages: list[dict], 
        add_generation_prompt: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        return_tensors: str = 'pt',
    ) -> int:

    image_processor = processor.image_processor
    merge_length = image_processor.merge_size ** 2
    temporal_patch_size = image_processor.temporal_patch_size
    patch_size = image_processor.patch_size

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
    len_input_ids = len(input_ids)
    return len_input_ids

        
def fast_fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Tuple[int, int]:
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



class PackingMapDataset:
    def __init__(
        self, 
        dataset, 
        processor, 
        data_args, 
        packing_workers: int = 4, 
        packing_interval: int = 64, 
        max_length: int = 4096, 
        shuffle_seed: int = 42,
        queue_maxsize: int = int(1e6)
    ):
        self.dataset = dataset
        self.processor = processor
        self.data_args = data_args
        self.packing_workers = packing_workers
        self.packing_interval = packing_interval
        self.max_length = max_length
        self.shuffle_seed = shuffle_seed
        self.queue_maxsize = queue_maxsize

    def _producer(self, indices, queue):
        for i in indices:
            messages = self.dataset.get_messages(i)
            len_input_ids = get_length_qwen2_5_vl_with_default_chat_template(
                self.processor, messages, 
                min_pixels=self.data_args.min_pixels, 
                max_pixels=self.data_args.max_pixels,
                )
            queue.put((i, messages, len_input_ids)) # indice, message, length
            # encoded = self.dataset.preprocess(messages, idx=i)
            # queue.put((i, len(encoded['input_ids'])))
        queue.put(None)  # 告知主进程本worker结束

    def cache_indices(self, save_path):
        # import ipdb; ipdb.set_trace()
        queue = mp.Queue(maxsize=self.queue_maxsize)
        indices = list(range(len(self.dataset)))
        shards = shard_dataset(indices, self.packing_workers, -1, self.shuffle_seed)
        
        workers = [
            mp.Process(target=self._producer, args=(shard, queue), daemon=True)
            for shard in shards
        ]
        for w in workers:
            w.start()

        buffer = []
        active_workers = self.packing_workers
        packed_data = []
        pbar = tqdm(total=len(indices), desc="Packing")

        while active_workers > 0:
            item = queue.get()
            if item is None:
                active_workers -= 1
                continue
            pbar.update(1)
            buffer.append(item) # add group
            if len(buffer) >= self.packing_interval:
                packed, buffer = self._pack(buffer)
                for group in packed:
                    packed_data.append([{"indice": i, "messages": msg, "length_input_ids": length} for i, msg, length in group])
        # flush remaining
        if buffer: # a single leftover group
            packed, _ = self._pack(buffer, final=True)
            for group in packed:
                packed_data.append([{"indice": i, "messages": msg, "length_input_ids": length} for i, msg, length in group])

        # with open(save_path, "w") as f:
        #     json.dump({"data": packed_data}, f, indent=2) # too slow (10min for dota_trainval512)
        with open(save_path, "w") as f:
            for group in packed_data:
                f.write(json.dumps(group) + "\n")  # 
        print(f"Saved {len(packed_data)} packed groups to {save_path}")

    def _pack(self, samples_and_lens, final=False):
        groups = binpacking.to_constant_volume(samples_and_lens, self.max_length, weight_pos=-1)
        if groups and not final:
            groups, leftover = groups[:-1], groups[-1]
        else:
            leftover = []
        return groups, leftover




class CachedPackingDatasetForQwen2_5VL(Dataset):
    def __init__(self, cache_path: str, processor, data_args, model_for_position_ids):
        self.cache_path = cache_path
        self.processor = processor
        self.data_args = data_args
        self.model_for_position_ids = model_for_position_ids

        self.min_pixels = self.data_args.min_pixels
        self.max_pixels = self.data_args.max_pixels
        self.patch_size = self.processor.image_processor.patch_size
        self.merge_size = self.processor.image_processor.merge_size
        self.image_factor = self.patch_size * self.merge_size

        with open(self.cache_path, "r") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        print(f'packed data loaded: {self.cache_path}')
        # cache['data'] ---> list of groups
        # cache['data'][group_i] ---> list of data dictn
        # cache['data'][group_i][data_i] ---> dict('indice': 123, 'message': a dict, 'length_input_ids': 123)

        # TODO: check with data_args, model_args

    def __len__(self):
        return len(self.data)

    def preprocess(self, messages, **kwargs):
        outputs = process_to_train_qwen2_5_vl_with_default_chat_template(
            self.processor, messages, 
            min_pixels=self.min_pixels, 
            max_pixels=self.max_pixels,
            model_for_position_ids=self.model_for_position_ids,
        )

        # Turning the first token of labels to -100 to prevent the last token of 
        # previous example predicting the first token of next example.
        # refer to https://github.com/huggingface/transformers/pull/31629
        outputs['labels'][0] = -100
        return outputs

    def __getitem__(self, index: int):
        list_grouped_data = self.data[index] # group
        outputs = []
        for data in list_grouped_data: # data dict
            messages = data['messages']
            outputs.append(self.preprocess(messages))
        # TODO: check with length, image_grid_thw
        return outputs


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)