import os
import json
import torch
import bisect
from collections import defaultdict

from .config import register_custom_sft_dataset, get_data_config_list, parse_config_file
from .sft_dataset import SupervisedDatasetForQwen2_5_VL
from .refgeo_dataset import GeoGroundDatasetForQwen2_5_VL, GeoGroundPolyOnlyDatasetForQwen2_5_VL, GeoGroundSegOnlyDatasetForQwen2_5_VL, GeoGroundWoSegDatasetForQwen2_5_VL, GeoGroundHbbOnlyDatasetForQwen2_5_VL
from .dense_det_dataset import DenseDetectionDatasetForQwen2_5_VL, DenseDetectionHbbOnlyDatasetForQwen2_5_VL, DenseDetectionPolyOnlyDatasetForQwen2_5_VL
from .visual_cot_dataset import VisualCOTDatasetForQwen2_5_VL
from .data_collator import DataCollatorForQwen2_5_VL, FlattenedDataCollatorForQwen2_5_VL
from .llava_onevision_dataset import LlavaOneVisionDatasetForQwen2_5_VL
from .xeochat_dataset import TeoChatDatasetForQwen2_5_VL
# from .data_packing import PackingMapDataset, CachedPackingDatasetForQwen2_5VL
from .fast_data_packing import PackingMapDataset, CachedPackingDatasetForQwen2_5VL

DATASET_CLASS = {
    "refgeo": GeoGroundDatasetForQwen2_5_VL,
    "refgeo_hbb_only": GeoGroundHbbOnlyDatasetForQwen2_5_VL,
    "refgeo_poly_only": GeoGroundPolyOnlyDatasetForQwen2_5_VL,
    "refgeo_seg_only": GeoGroundSegOnlyDatasetForQwen2_5_VL,
    "refgeo_wo_seg": GeoGroundWoSegDatasetForQwen2_5_VL,
    "supervised": SupervisedDatasetForQwen2_5_VL,
    "dense_det": DenseDetectionDatasetForQwen2_5_VL, 
    "dense_det_hbb_only": DenseDetectionHbbOnlyDatasetForQwen2_5_VL,
    "dense_det_poly_only": DenseDetectionPolyOnlyDatasetForQwen2_5_VL,
    "visual_cot": VisualCOTDatasetForQwen2_5_VL,
    "llava_onevision": LlavaOneVisionDatasetForQwen2_5_VL,
    "teochat": TeoChatDatasetForQwen2_5_VL,
}


class ConcatDatasetForQwen2_5_VL(torch.utils.data.ConcatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(self)

    def get_idx(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_messages(self, idx):
        dataset_idx, sample_idx = self.get_idx(idx)
        return self.datasets[dataset_idx].get_messages(sample_idx)

    def preprocess(self, messages, *, idx=None, **kwargs):
        dataset_idx, _ = self.get_idx(idx)
        return self.datasets[dataset_idx].preprocess(messages)

    def __repr__(self):
        dataset_type_sample_num = {}
        dataset_info = {}
        for dataset in self.datasets:
            if hasattr(dataset, 'data_config_list'):
                dataset_type = dataset.data_config_list[0].get('dataset_type', 'sft')
                dataset_type_sample_num[dataset_type] = f"len: {len(dataset)} percent: {len(dataset) / len(self) * 100:.2f}%"
                dataset_info[dataset_type] = {
                    'num_samples': len(dataset),
                    'data_config_list': dataset.data_config_list,
                }
        return f"# dataset_type_sample_num: \n```json\n{json.dumps(dataset_type_sample_num, indent=4)}\n```\n" + \
               f"# dataset_info: \n```json\n{json.dumps(dataset_info, indent=4)}\n```\n"


def make_supervised_data_module(processor, data_args, model_for_position_ids=None):
    """Make dataset and collator for supervised fine-tuning."""
    datasets_param = data_args.datasets

    # old api (only support for SupervisedDatasetForQwen2_5_VL)
    data_path = getattr(data_args, 'data_path', None)
    image_folder = getattr(data_args, 'image_folder', None)
    if data_path is not None or image_folder is not None:
        assert data_path is not None and image_folder is not None and datasets_param is None, data_args
        datasets_param = register_custom_sft_dataset(data_path, image_folder)

    # maybe use a json/yaml config file
    if isinstance(datasets_param, str):
        if datasets_param.endswith(".json") or datasets_param.endswith(".yaml"):
            datasets_param = parse_config_file(datasets_param)
        else:
            datasets_param = [datasets_param]

    # get configs
    data_config_list = get_data_config_list(datasets_param)
    type_data_config_list = defaultdict(list)
    for data_config in data_config_list:
        type_data_config_list[data_config.get("dataset_type", 'supervised')].append(data_config)

    # prepare train dataset
    datasets_to_concat = []
    for dataset_type, data_config_list in type_data_config_list.items():
        datasets_to_concat.append(DATASET_CLASS[dataset_type](
            data_config_list, processor=processor, data_args=data_args, 
            model_for_position_ids=model_for_position_ids))
        datasets_to_concat[-1].data_config_list = data_config_list
    if len(datasets_to_concat) > 1:
        train_dataset = ConcatDatasetForQwen2_5_VL(datasets_to_concat)
    else:
        train_dataset = datasets_to_concat[0]

    # prepare packing data
    if getattr(data_args, 'packing_data', False):
        train_dataset = PackingMapDataset(
            train_dataset, 
            processor, data_args,
            packing_workers=getattr(data_args, "packing_workers", 32),
            packing_interval=getattr(data_args, "packing_interval", 64),
            max_length=getattr(data_args, "max_length", 4096),
            shuffle_seed=getattr(data_args, "dataset_randomness", 42),
        )
        # cache packed data
        if not os.path.exists(data_args.packing_cache):
            os.makedirs(os.path.dirname(data_args.packing_cache), exist_ok=True)
            # data packing and cache
            train_dataset.cache_indices(save_path=data_args.packing_cache)

    if getattr(data_args, 'packing_data', False) and getattr(data_args, 'packing_cache', None) is not None:
        train_dataset = CachedPackingDatasetForQwen2_5VL(data_args.packing_cache, processor, data_args, model_for_position_ids)
    # prepare data collator
    if getattr(data_args, 'flatten_data', False):
        data_collator = FlattenedDataCollatorForQwen2_5_VL(max_length=getattr(data_args, 'max_length', -100), process_exceed=getattr(data_args, 'process_exceed', 'truncate'))
    else:
        data_collator = DataCollatorForQwen2_5_VL(pad_token_id=processor.tokenizer.pad_token_id, max_length=getattr(data_args, 'max_length', -100), process_exceed=getattr(data_args, 'process_exceed', 'truncate'))

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
