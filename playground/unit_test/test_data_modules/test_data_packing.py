import re
import torch
from argparse import Namespace

from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from rscovlm.training.data import make_supervised_data_module

from PIL import Image
Image.MAX_IMAGE_PIXELS = 2e8  # 根据你的需求设置更大的限制

CHECK_LOSS = False

model_path = "./playground/Qwen2.5-VL-3B-Instruct"
data_args = Namespace(
    datasets=["vhm_dataset%1", "refgeo%1"],
    flatten_data=True,
    data_path=None,
    image_folder=None,
    min_pixels=256 * 28 * 28,  # 4 * 28 * 28
    max_pixels=1296 * 28 * 28,  # 16384 * 28 * 28
    packing_data=True,
    packing_workers=16,
    packing_interval=32,
    packing_shuffle_seed=42,
    max_length=4096,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

if CHECK_LOSS:
    model = model.cuda().eval()
import ipdb; ipdb.set_trace()
modules = make_supervised_data_module(processor, data_args, model)
train_dataset, data_collator = modules["train_dataset"], modules["data_collator"]

# # test cache dataset
# # heavy cache
# train_dataset.cache_all(save_path="./tmp_packing/", cache_dir="./tmp_hf_cache", max_shard_size="500MB", writer_batch_size=8)
# light cache
import ipdb; ipdb.set_trace()
train_dataset.cache_indices(save_path="./debug_tmp_packing_indices.json")

# from torch.utils.data import DataLoader
# train_dataloder = DataLoader(train_dataset, batch_size=1, collate_fn=data_collator, num_workers=8)
# train_dataiter = iter(train_dataloder)

# sample = next(train_dataiter)
# input_ids = sample["input_ids"]
# labels = sample["labels"]
# position_ids = sample["position_ids"]
# pixel_values = sample["pixel_values"]
# image_grid_thw = sample["image_grid_thw"]
# # import ipdb; ipdb.set_trace()

# # 测试重制资源
# train_dataset.reset_resource()

# def simplify_output(text):
#     # return text
#     if isinstance(text, list):
#         return [simplify_output(t) for t in text]
#     patterns = [r'<\|image_pad\|>', r'<\|endoftext\|>', r'!']
#     for pattern in patterns:
#         text = re.sub(rf'({pattern})\1+', lambda m: f"{m.group(1)} * {len(m.group(0)) // len(m.group(1))} ", text)
#     return text

# from copy import deepcopy
# from colorama import init, Fore, Back, Style
# init(autoreset=True)
# def print_with_highlight(text, keywords: list[str]):
#     def _process(_text, keywords):
#         # _text = deepcopy(_text)
#         # if isinstance(keywords, str):
#         #     keywords = [keywords]
#         # for keyword in keywords:
#         #     _text = _text.replace(keyword, f"{Back.YELLOW}{Fore.BLACK}{keyword}{Style.RESET_ALL}")
#         return _text
#     if isinstance(text, list):
#         for t in text:  
#             print(_process(t, keywords))
#     else:
#         print(text)

# # 检查loss和是否对应模型接口
# losses = []
# for _ in range(16):
#     batch = next(train_dataiter)

#     input_ids = batch['input_ids']
#     pixel_values = batch['pixel_values']
#     image_grid_thw = batch['image_grid_thw']
#     labels = batch['labels']
#     position_ids = batch['position_ids']

#     print(); print(f'position_ids: ')
#     pos = position_ids[:, 0]  # shape: [3, L]
#     text_mask = (pos[0] == pos[1]) & (pos[1] == pos[2])
#     image_mask = ~text_mask
#     print("text_length:", text_mask.sum().item())
#     print("image token numbers:", image_mask.sum().item())
#     print("image t values:", pos[0][image_mask].unique())
#     print("image h values:", pos[1][image_mask].unique())
#     print("image w values:", pos[2][image_mask].unique())
#     # sanity check
#     print("position_ids shape:", batch["position_ids"].shape); print()

#     print("input_ids_decoded")
#     input_ids_decoded = processor.tokenizer.batch_decode(input_ids)
#     print_with_highlight(simplify_output(input_ids_decoded), '<im_start>system'); print()

#     print("labels_decoded")
#     converted_labels = []
#     for label in labels:
#         converted_label = label.clone()
#         converted_label[converted_label == -100] = 0
#         converted_labels.append(converted_label)
#     converted_labels = [label.tolist() for label in converted_labels]
#     labels_decoded = processor.tokenizer.batch_decode(converted_labels)
#     print_with_highlight(simplify_output(labels_decoded), '<|im_end|>'); print()

#     print("pure_labels_decoded")
#     pure_labels = [label[label != -100].tolist() for label in labels]
#     pure_labels_decoded = processor.tokenizer.batch_decode(pure_labels)
#     print_with_highlight(simplify_output(pure_labels_decoded), '<|im_end|>'); print()

#     if CHECK_LOSS:
#         batch_on_device = {}
#         for key, value in batch.items():
#             if isinstance(value, torch.Tensor):
#                 batch_on_device[key] = value.cuda().to(torch.bfloat16) if value.dtype == torch.float else value.cuda()
#             else:
#                 batch_on_device[key] = value
        
#         naive_position_ids = batch_on_device.pop('naive_position_ids')
#         from rscovlm.training.monkey_patch.varlen import monkey_patch_flash_attention_to_pass_position_ids
#         monkey_patch_flash_attention_to_pass_position_ids(naive_position_ids)
#         with torch.no_grad():
#             loss = model(**batch_on_device).loss
#         losses.append(loss)
#         print(f'losses: {losses}')
    
# import ipdb; ipdb.set_trace()
