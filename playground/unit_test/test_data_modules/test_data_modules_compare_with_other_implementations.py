import re
import torch
import random
import inspect
from argparse import Namespace
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from rscovlm.training.data.config import data_dict
from rscovlm.training.data import make_supervised_data_module
from legacy_dataset import make_supervised_data_module as make_supervised_data_module_legacy
from qwen_official import make_supervised_data_module as make_supervised_data_module_official


model_path = "./playground/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

def get_rope_index(**kwargs):
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

processor.get_rope_index = get_rope_index

data_args_latest = Namespace(
    datasets=["vhm_dataset"],
    data_flatten=False,
    data_path=None,
    image_folder=None,
    min_pixels=256 * 28 * 28,  # 4 * 28 * 28
    max_pixels=1296 * 28 * 28,  # 16384 * 28 * 28
    video_min_pixels=128 * 28 * 28,  # 128 * 28 * 28
    video_max_pixels=768 * 28 * 28,  # 768 * 28 * 28
)
modules = make_supervised_data_module(processor, data_args_latest, model)
train_dataset, data_collator = modules["train_dataset"], modules["data_collator"]

data_args_legacy = Namespace(
    dataset_type=["sft"],
    data_path=[data_dict["vhm_dataset"]["annotation_path"]],
    image_folder=[data_dict["vhm_dataset"]["data_path"]],
    min_pixels=256 * 28 * 28,  # 4 * 28 * 28
    max_pixels=1296 * 28 * 28,  # 16384 * 28 * 28
    video_min_pixels=128 * 28 * 28,  # 128 * 28 * 28
    video_max_pixels=768 * 28 * 28,  # 768 * 28 * 28
)

modules_legacy = make_supervised_data_module_legacy(model_path, processor, data_args_legacy)
train_dataset_legacy, data_collator_legacy = modules_legacy["train_dataset"], modules_legacy["data_collator"]

data_args_official = Namespace(
    dataset_use="vhm_dataset",
    data_flatten=False,
    min_pixels=256 * 28 * 28,  # 4 * 28 * 28
    max_pixels=1296 * 28 * 28,  # 16384 * 28 * 28
    video_min_pixels=128 * 28 * 28,  # 128 * 28 * 28
    video_max_pixels=768 * 28 * 28,  # 768 * 28 * 28
    model_type = "qwen2.5vl",
    image_processor=processor.image_processor,
)
modules_official = make_supervised_data_module_official(processor.tokenizer, data_args_official)
train_dataset_official, data_collator_official = modules_official["train_dataset"], modules_official["data_collator"]

# 获取一个样本，debug数据集的get item
example = train_dataset[3]
example_legacy = train_dataset_legacy[3]
example_official = train_dataset_official[3]

from torch.utils.data import DataLoader
train_dataloder = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator, num_workers=0, shuffle=False)
train_dataiter = iter(train_dataloder)
train_dataloder_legacy = DataLoader(train_dataset_legacy, batch_size=2, collate_fn=data_collator_legacy, num_workers=0, shuffle=False)
train_dataiter_legacy = iter(train_dataloder_legacy)
train_dataloder_official = DataLoader(train_dataset_official, batch_size=2, collate_fn=data_collator_official, num_workers=0, shuffle=False)
train_dataiter_official = iter(train_dataloder_official)

def simplify_output(text):
    # return text
    if isinstance(text, list):
        return [simplify_output(t) for t in text]
    patterns = [r'<\|image_pad\|>', r'<\|endoftext\|>', r'!']
    for pattern in patterns:
        text = re.sub(rf'({pattern})\1+', lambda m: f"{m.group(1)} * {len(m.group(0)) // len(m.group(1))} ", text)
    return text

# 检查loss和是否对应模型接口
losses = []
for _ in range(2):
    batch = next(train_dataiter)
    batch_legacy = next(train_dataiter_legacy)
    batch_official = {k:v for k, v in next(train_dataiter_official).items() if v is not None}

    input_ids = batch['input_ids']
    input_ids_legacy = batch_legacy['input_ids']
    input_ids_official = batch_official['input_ids']

    assert torch.equal(input_ids, input_ids_legacy)

    input_ids_decoded = "".join(processor.tokenizer.batch_decode(input_ids))
    input_ids_official_decoded = "".join(processor.tokenizer.batch_decode(input_ids_official))
    num_image_pad = input_ids_decoded.count("<|image_pad|>")
    num_image_pad_official = input_ids_official_decoded.count("<|image_pad|>")
    input_ids_decoded = input_ids_decoded.replace("<|image_pad|>", " ")
    input_ids_official_decoded = input_ids_official_decoded.replace("<|image_pad|>", " ")
    assert num_image_pad == num_image_pad_official, f"num_image_pad: {num_image_pad}, num_image_pad_official: {num_image_pad_official}"
    assert input_ids_decoded == input_ids_official_decoded.replace("<|vision_end|>\n", "<|vision_end|>")
    
    labels = batch['labels']
    labels_legacy = batch_legacy['labels']
    labels_official = batch_official['labels']

    assert (labels == labels_legacy).sum() == labels.shape[1] * 2 - 2
    labels = [label[label != -100] for label in labels]
    labels_legacy = [label[label != -100] for label in labels_legacy]
    labels_official = [label[label != -100] for label in labels_official]
    assert all(torch.equal(l1, l2) for l1, l2 in zip(labels_legacy, labels_official))
    assert all(torch.equal(l1[:-1], l2) for l1, l2 in zip(labels_legacy, labels))

    attention_mask = batch['attention_mask'].bool()
    attention_mask_legacy = batch_legacy['attention_mask']
    attention_mask_official = batch_official['attention_mask']

    assert torch.equal(attention_mask, attention_mask_legacy)

    pixel_values = batch['pixel_values']
    pixel_values_legacy = batch_legacy['pixel_values']
    pixel_values_official = batch_official['pixel_values']

    assert torch.equal(pixel_values, pixel_values_legacy)
    assert torch.equal(pixel_values, pixel_values_official)

    image_grid_thw = batch['image_grid_thw']
    image_grid_thw_legacy = batch_legacy['image_grid_thw']
    image_grid_thw_official = batch_official['image_grid_thw']

    assert torch.equal(image_grid_thw, image_grid_thw_legacy)
    assert torch.equal(image_grid_thw, image_grid_thw_official)

    position_ids = batch['position_ids']
    position_ids_legacy = None
    position_ids_official = batch_official['position_ids']

    import ipdb; ipdb.set_trace()
