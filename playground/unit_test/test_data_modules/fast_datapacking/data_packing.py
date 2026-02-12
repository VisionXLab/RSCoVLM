# We follow the [implementations of ms-swift](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/utils.py)
import json
import random
import warnings
import binpacking
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Subset, get_worker_info, IterableDataset

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


def concat_grouped_examples(samples_group: list[dict[str, any]]) -> dict[str, any]:
    grouped_input_ids = []
    grouped_labels = []
    grouped_position_ids = []
    grouped_pixel_values = []
    grouped_image_grid_thw = []
    grouped_data_indices = []
    for sample in samples_group:
        grouped_input_ids.append(sample['input_ids'])
        grouped_labels.append(sample['labels'])
        grouped_position_ids.append(sample['position_ids'])
        grouped_pixel_values.append(sample['pixel_values'])
        grouped_image_grid_thw.append(sample['image_grid_thw'])
        if 'indice' in sample:
            grouped_data_indices.append(sample['indice'])
        assert sum(sample['attention_mask']) == len(sample['attention_mask'])
    if isinstance(sample['input_ids'], np.ndarray):
        output = {
            'input_ids': np.concatenate(grouped_input_ids, axis=0).astype(np.int32),
            'labels': np.concatenate(grouped_labels, axis=0).astype(np.int32),
            'position_ids': np.concatenate(grouped_position_ids, axis=1).astype(np.int16),
            'pixel_values': np.concatenate(grouped_pixel_values, axis=0),
            'image_grid_thw': np.concatenate(grouped_image_grid_thw, axis=0),
        }
    else:
        output = {
            'input_ids': torch.cat(grouped_input_ids, dim=0),
            'labels': torch.cat(grouped_labels, dim=0),
            'position_ids': torch.cat(grouped_position_ids, dim=1),
            'pixel_values': torch.cat(grouped_pixel_values, dim=0),
            'image_grid_thw': torch.cat(grouped_image_grid_thw, dim=0),
        }
    if len(grouped_data_indices) > 0:
        output['indices'] = grouped_data_indices
    return output


def calculate_matched_group(
    list_samples_and_length: list[tuple[dict[str, any], int]], 
    is_finished: bool = True, 
    max_length: int = 8192,
    output_concated_examples: bool = False
) -> tuple[list[dict[str, any]], list[tuple[dict[str, any], int]]]:
    if len(list_samples_and_length) == 0:
        return [], []
    list_samples_and_length = binpacking.to_constant_volume(list_samples_and_length,  max_length, weight_pos=1)
    packed_samples = []
    if list_samples_and_length and not is_finished:
        # refer to [Fewer Truncations Improve Language Modeling](https://arxiv.org/pdf/2404.10830)
        list_samples_and_length, rest_list_samples_and_length = list_samples_and_length[:-1], list_samples_and_length[-1]
    else:
        rest_list_samples_and_length = []
    for row in list_samples_and_length:
        _row = [r[0] for r in row]
        if output_concated_examples:
            packed_samples.append(concat_grouped_examples(_row))
        else:
            packed_samples.append(_row)
    return packed_samples, rest_list_samples_and_length


class PackingMapDataset(object):

    def __new__(cls, *args, **kwargs):
        if kwargs.get('streaming', True):
            # For streaming mode, we set the dataset as an `IterableDataset`.
            cls = type('IterablePackingMapDataset', (PackingMapDataset, IterableDataset), {})
        else:
            # For non-streaming mode, we set the dataset as a `Dataset`.
            cls = type('PackingMapDataset', (PackingMapDataset, Dataset), {})
        return super().__new__(cls)

    def __init__(
        self, 
        dataset: Dataset, 
        *, 
        packing_workers: int = 1, 
        packing_interval: int = 64, 
        max_length: int = 4096,
        shuffle_seed: Optional[int] = None,
        subprocess_num_threads: int = 1,
        streaming: bool = True,
        queue_maxsize: int = 1e6,
        lazy: bool = False,
    ):
        self.dataset = dataset
        self.packing_workers = packing_workers
        self.packing_interval = packing_interval
        self.max_length = max_length
        self.shuffle_seed = shuffle_seed
        self.streaming = streaming
        self.queue_maxsize = queue_maxsize
        self.subprocess_num_threads = subprocess_num_threads
        if subprocess_num_threads > 1:
            warnings.warn("subprocess_num_threads > 1 may cause deadlock.")
        self._return_indice = False
        self._return_type = 'pt'
        self._use_manager = True
        self._output_concated_examples = False
        self._return_messages = False

        if not lazy:
            self.reset_resource()

    def reset_resource(self, use_manager: Optional[bool] = None):
        self.terminate_resource()

        data_indices_shards = self.reset_indices()
        if dist.is_initialized():
            bar_disable = dist.get_rank() != 0
        else:
            bar_disable = False
        # bar_disable = True
        tqdm_total = sum(len(_) for _ in data_indices_shards)
        self.prog_bar = tqdm(total=tqdm_total, dynamic_ncols=True, desc='Packing', disable=bar_disable)

        # We should use manager when using datasets.Dataset.from_generator 
        # because only manager queue can be pickled. FIXME in future.
        use_manager = use_manager if use_manager is not None else self._use_manager
        self._queue = mp.Manager().Queue(maxsize=self.queue_maxsize) if use_manager else mp.Queue(maxsize=self.queue_maxsize)
        
        data_indices_shards = self.reset_indices()

        self.workers = []
        self._terminated_workers = 0
        for i in range(self.packing_workers):
            worker = mp.Process(target=self._producer, args=(self.dataset, data_indices_shards[i], self._queue), daemon=True)
            worker.start()
            self.workers.append(worker)

        if not self.streaming:
            try:
                self.packed_dataset = list(self.get_packed_samples())
            finally:
                self.terminate_resource()

    def terminate_resource(self):
        if hasattr(self, 'prog_bar'):
            self.prog_bar.close()
            del self.prog_bar
        if hasattr(self, 'workers'):
            for worker in self.workers:
                worker.terminate()
            del self.workers
        if hasattr(self, '_queue'):
            del self._queue
        if hasattr(self, '_terminated_workers'):
            del self._terminated_workers

    def reset_indices(self, shuffle_seed: Optional[int] = None):
        if shuffle_seed is not None:
            self.shuffle_seed = shuffle_seed
        
        data_indices = list(range(len(self.dataset)))

        if self.streaming:
            # For streaming mode, it should be an iterable dataset. 
            # We need to shard the dataset for each worker&rank by hand.
            if dist.is_initialized():
                dist_rank = dist.get_rank()
                dist_world_size = dist.get_world_size()
            else:
                dist_rank = 0
                dist_world_size = 1

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                dataloader_worker_id = worker_info.id
                dataloader_num_workers = worker_info.num_workers
            else:
                dataloader_worker_id = 0
                dataloader_num_workers = 1

            shard_id = dist_rank * dataloader_num_workers + dataloader_worker_id
            n_shards = dist_world_size * dataloader_num_workers
            data_indices = shard_dataset(data_indices, n_shards, shard_id, self.shuffle_seed)

        data_indices_shards = shard_dataset(data_indices, self.packing_workers, -1, self.shuffle_seed)
        return data_indices_shards

    def fetch_packing_data(self, prev_data: Optional[list] = None) -> list[tuple[dict[str, any], int]]:
        data = prev_data or []
        for _ in range(self.packing_interval):
            encoded_data = self._queue.get()
            if encoded_data is None:
                self._terminated_workers += 1
                if self._terminated_workers == self.packing_workers:
                    break
                continue
            self.prog_bar.update(1)
            if encoded_data:
                data.append((encoded_data, len(encoded_data['input_ids'])))
        return data
    
    def get_packed_samples(self, output_concated_examples: Optional[bool] = None) -> list[dict[str, any]]:
        output_concated_examples = output_concated_examples if output_concated_examples is not None else self._output_concated_examples
        data = []
        while True:
            data = self.fetch_packing_data(data)
            is_finished = self._terminated_workers == self.packing_workers
            packed_samples, data = calculate_matched_group(  # packed_samples, rest_list_samples_and_length
                data, 
                is_finished=is_finished,
                max_length=self.max_length, 
                output_concated_examples=output_concated_examples,
            )
            yield from packed_samples
            if is_finished:
                break
    
    def set_torch_threads(self):
        # print(f"packing subprocess: os.cpu_count()={os.cpu_count()}", flush=True)
        # print(f"packing subprocess: setting threads {torch.get_num_threads()} to {self.subprocess_num_threads}", flush=True)
        torch.set_num_threads(self.subprocess_num_threads)

    def _producer(
        self, dataset: Dataset, 
        dataset_indices_shard: list[int], 
        queue: mp.Queue, 
        return_indice: Optional[bool] = None, 
        return_type: Optional[str] = None,
        return_messages: Optional[bool] = None,
    ):
        return_indice = return_indice if return_indice is not None else self._return_indice
        return_type = return_type if return_type is not None else self._return_type
        return_messages = return_messages if return_messages is not None else self._return_messages
        self.set_torch_threads()
        for i in dataset_indices_shard:
            messages = dataset.get_messages(i)
            encoded_data = dataset.preprocess(messages, idx=i)

            # Turning the first token of labels to -100 to prevent the last token of 
            # previous example predicting the first token of next example.
            # refer to https://github.com/huggingface/transformers/pull/31629
            encoded_data['labels'][0] = -100

            if return_type == 'np':
                for k, v in encoded_data.items():
                    if isinstance(v, torch.Tensor):
                        encoded_data[k] = v.numpy()
            elif return_type == 'list':
                for k, v in encoded_data.items():
                    if isinstance(v, torch.Tensor):
                        encoded_data[k] = v.tolist()
            else:
                assert return_type == 'pt', f"return_type should be 'np' or 'pt', but got {return_type}"
            
            if return_indice:
                encoded_data['indice'] = i

            if return_messages:
                encoded_data['messages'] = messages
            
            queue.put(encoded_data)
        queue.put(None)

    def __getitem__(self, index: int):
        if self.streaming:
            raise TypeError("For streaming mode, you should use this dataset as an iterable. Please use iter(dataset) to get samples.")
        if getattr(self, 'packed_dataset', None) is None:
            self.reset_resource()
        return self.packed_dataset[index].copy()

    def __len__(self):
        if self.streaming:
            raise TypeError("For streaming mode, you should use this dataset as an iterable, not by length.")
        if getattr(self, 'packed_dataset', None) is None:
            self.reset_resource()
        return len(self.packed_dataset)
    
    def __iter__(self):
        try:
            if self.streaming:
                if len(getattr(self, 'workers', [])) == 0:
                    self.reset_resource()
                data_generator = self.get_packed_samples()
                yield from data_generator
            else:
                if getattr(self, 'packed_dataset', None) is None:
                    self.reset_resource()
                yield from self.packed_dataset
        finally:
            self.terminate_resource()

    def cache_all(
        self, 
        save_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        writer_batch_size: int = 8,
        max_shard_size: str = "500MB",
    ):
        prev_return_type, self._return_type = self._return_type, 'np'
        prev_output_concated_examples, self._output_concated_examples = self._output_concated_examples, True

        self.reset_resource()

        def data_gen():
            packed_sample_length = []
            for grouped_examples in iter(self):
                grouped_examples['input_ids'] = grouped_examples['input_ids'].astype(np.int32)
                grouped_examples['labels'] = grouped_examples['labels'].astype(np.int32)
                if 'position_ids' in grouped_examples:
                    grouped_examples['position_ids'] = grouped_examples['position_ids'].astype(np.int16)
                packed_sample_length.append(len(grouped_examples['input_ids']))
                yield grouped_examples
            packed_sample_length.sort()
            print(f"packed_sample_length[:10]: {packed_sample_length[:10]}", flush=True)
            print(f"packed_sample_length[-10:]: {packed_sample_length[-10:]}", flush=True)
        
        from datasets import Dataset
        try:
            ds = Dataset.from_generator(data_gen, cache_dir=cache_dir, writer_batch_size=writer_batch_size)
            print(f"ds: {repr(ds)}", flush=True)
            if save_path is not None:
                ds.save_to_disk(save_path, max_shard_size=max_shard_size)
        finally:
            self.terminate_resource()
        
        self._return_type = prev_return_type
        self._output_concated_examples = prev_output_concated_examples
        return ds

    def cache_indices(self, save_path: Optional[str] = None):
        prev_return_indice, self._return_indice = self._return_indice, True
        prev_return_type, self._return_type = self._return_type, 'np'
        prev_output_concated_examples, self._output_concated_examples = self._output_concated_examples, True

        self.reset_resource()

        def data_gen():
            packed_sample_length = []
            for grouped_examples in iter(self):
                packed_sample_length.append(len(grouped_examples['input_ids']))
                yield grouped_examples['indices']
            packed_sample_length.sort()
            print(f"packed_sample_length[:10]: {packed_sample_length[:10]}", flush=True)
            print(f"packed_sample_length[-10:]: {packed_sample_length[-10:]}", flush=True)

        try:
            indices_cache = list(data_gen())
            print(f"len(packed_sample) = {len(indices_cache)}", flush=True)
        finally:
            self.terminate_resource()

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(indices_cache, f)

        self._return_indice = prev_return_indice
        self._return_type = prev_return_type
        self._output_concated_examples = prev_output_concated_examples
        return indices_cache

    def cache_messages(self, save_path: Optional[str] = None, return_length: bool = False):
        prev_output_concated_examples, self._output_concated_examples = self._output_concated_examples, False
        prev_return_messages, self._return_messages = self._return_messages, True

        self.reset_resource()

        def data_gen():
            packed_sample_length = []
            for grouped_examples in iter(self):
                grouped_messages = []
                grouped_examples_length = 0
                for example in grouped_examples:
                    grouped_examples_length += len(example['input_ids'])
                    if return_length:
                        image_grid_thw = example['image_grid_thw']
                        if not isinstance(image_grid_thw, list):
                            image_grid_thw = image_grid_thw.tolist()
                        grouped_messages.append((example['messages'], len(example['input_ids']), image_grid_thw))
                    else:
                        grouped_messages.append(example['messages'])
                packed_sample_length.append(grouped_examples_length)
                yield grouped_messages
            packed_sample_length.sort()
            print(f"packed_sample_length[:10]: {packed_sample_length[:10]}", flush=True)
            print(f"packed_sample_length[-10:]: {packed_sample_length[-10:]}", flush=True)

        try:
            messages_cache = list(data_gen())
            print(f"len(packed_sample) = {len(messages_cache)}", flush=True)
        finally:
            self.terminate_resource()

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(messages_cache, f)

        self._output_concated_examples = prev_output_concated_examples
        self._return_messages = prev_return_messages
        return messages_cache


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

        with open(cache_path, "r") as f:
            cache = json.load(f)
        self.data = cache['data']

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
        list_grouped_data = self.data[index]
        outputs = []
        for data in list_grouped_data:
            messages = data['messages']
            outputs.append(self.preprocess(messages))
        # TODO: check with length, image_grid_thw
        return outputs


def prepacking_dataset():
    from PIL import Image
    from dataclasses import asdict, dataclass, field
    from rscovlm.training.data import make_supervised_data_module
    from rscovlm.training.params import DataArguments, ModelArguments, HfArgumentParser
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    Image.MAX_IMAGE_PIXELS = 8e8
    @dataclass
    class CustomArguments:
        save_path: str = field(default=None, metadata={"help": "Path to save the packed dataset."})

    parser = HfArgumentParser((DataArguments, ModelArguments, CustomArguments))
    data_args, model_args, args = parser.parse_args_into_dataclasses()

    if data_args.datasets is not None and len(data_args.datasets) == 1:
        data_args.datasets = data_args.datasets[0]
    if data_args.data_path is not None and len(data_args.data_path) == 1:
        data_args.data_path = data_args.data_path[0]
    if data_args.image_folder is not None and len(data_args.image_folder) == 1:
        data_args.image_folder = data_args.image_folder[0]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_id, torch_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_id, use_fast=True)

    modules = make_supervised_data_module(processor, data_args, model)
    train_dataset = modules["train_dataset"]

    messages_and_length_cache = train_dataset.cache_messages(return_length=True)
    data = []
    for grouped_messages in messages_and_length_cache:
        grouped = []
        for messages, encoded_length, image_grid_thw in grouped_messages:
            grouped.append(dict(messages=messages, encoded_length=encoded_length, image_grid_thw=image_grid_thw))
        data.append(grouped)
    
    data = {'data_args': asdict(data_args), 'model_args': asdict(model_args), 'data': data}
    print(f"Saving packed dataset to {args.save_path}", flush=True)
    with open(args.save_path, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    prepacking_dataset()
