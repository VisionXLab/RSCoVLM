import re
import os
import torch
import random
import inspect
from tqdm import tqdm
from argparse import Namespace
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from rscovlm.training.data import make_supervised_data_module

CHECK_LOSS = False
flatten_data = False

if flatten_data:
    from rscovlm.training.monkey_patch.flatten_data import replace_qwen2_vl_attention_class
    replace_qwen2_vl_attention_class()

model_path = "./playground/Qwen2.5-VL-3B-Instruct"
datasets = ["vhm_dataset%50"] # 

data_args = Namespace(
    datasets=datasets,
    flatten_data=flatten_data,
    data_path=None,
    image_folder=None,
    min_pixels=256 * 28 * 28,  # 4 * 28 * 28
    max_pixels=1296 * 28 * 28,  # 16384 * 28 * 28
    video_min_pixels=128 * 28 * 28,  # 128 * 28 * 28
    video_max_pixels=768 * 28 * 28,  # 768 * 28 * 28
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

if CHECK_LOSS:
    model = model.cuda().eval()

modules = make_supervised_data_module(processor, data_args, model)
train_dataset, data_collator = modules["train_dataset"], modules["data_collator"]
print(f'total train data: {len(train_dataset)}')
# 获取一个样本，debug数据集的get item
import ipdb; ipdb.set_trace()
example = train_dataset[3]

# merge data
# all_meta_with_dataset = []
# for dataset in train_dataset.datasets:
#     if hasattr(dataset, 'list_data_dict'):
#         for meta in dataset.list_data_dict:
#             all_meta_with_dataset.append((meta, dataset))
#     elif hasattr(dataset, 'list_data_meta'):
#         for meta in dataset.list_data_meta:
#             all_meta_with_dataset.append((meta, dataset))

# sampled_data = random.sample(all_meta_with_dataset, 50)
# for meta, source_dataset in sampled_data:
#     # only grounding supports output resize info
#     if hasattr(source_dataset, 'add_resized_size_to_metainfo'):
#         print(f"Processing with dataset: {type(source_dataset).__name__}")
#         source_dataset.add_resized_size_to_metainfo(meta)
#         w, h = meta['image_width'], meta['image_height']
#         wr, hr = meta.get('resized_width', 'N/A'), meta.get('resized_height', 'N/A')
#         print(f"image size: {w}x{h} -> {wr}x{hr}")
#     else:
#         pass
#         # print(f"Skipping - dataset {type(source_dataset).__name__} doesn't support resizing_info")

# from torch.utils.data import DataLoader
# train_dataloder = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator, num_workers=0, shuffle=True)
# train_dataiter = iter(train_dataloder)

# # 怎么拼成一个batch，debug数据集的collate_fn/datacollator
# example_batch = next(train_dataiter)

# def simplify_output(text):
#     # return text
#     if isinstance(text, list):
#         return [simplify_output(t) for t in text]
#     patterns = [r'<\|image_pad\|>', r'<\|endoftext\|>', r'!']
#     for pattern in patterns:
#         text = re.sub(rf'({pattern})\1+', lambda m: f"{m.group(1)} * {len(m.group(0)) // len(m.group(1))} ", text)
#     return text

# # 检查loss和是否对应模型接口
# losses = []
# for _ in range(2):
#     batch = next(train_dataiter)

#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     pixel_values = batch['pixel_values']
#     image_grid_thw = batch['image_grid_thw']
#     labels = batch['labels']
#     position_ids = batch['position_ids']

#     print(f'position_ids: ')
#     pos = position_ids[:, 0]  # shape: [3, L]
#     text_mask = (pos[0] == pos[1]) & (pos[1] == pos[2])
#     image_mask = ~text_mask
#     print("text_length:", text_mask.sum().item())
#     print("image token numbers:", image_mask.sum().item())
#     print("image t values:", pos[0][image_mask].unique())
#     print("image h values:", pos[1][image_mask].unique())
#     print("image w values:", pos[2][image_mask].unique())
#     # sanity check
#     print("position_ids shape:", batch["position_ids"].shape)

#     print("input_ids_decoded")
#     input_ids_decoded = processor.tokenizer.batch_decode(input_ids)
#     print(simplify_output(input_ids_decoded)); print()

#     print("labels_decoded")
#     converted_labels = []
#     for label in labels:
#         converted_label = label.clone()
#         converted_label[converted_label == -100] = 0
#         converted_labels.append(converted_label)
#     converted_labels = [label.tolist() for label in converted_labels]
#     labels_decoded = processor.tokenizer.batch_decode(converted_labels)
#     print(simplify_output(labels_decoded)); print()

#     print("pure_labels_decoded")
#     pure_labels = [label[label != -100].tolist() for label in labels]
#     pure_labels_decoded = processor.tokenizer.batch_decode(pure_labels)
#     print(simplify_output(pure_labels_decoded)); print()

#     if not flatten_data:
#         print("length wo padding")
#         print(attention_mask.sum(dim=1)); print()

#     if CHECK_LOSS:
#         batch_on_device = {}
#         for key, value in batch.items():
#             if isinstance(value, torch.Tensor):
#                 batch_on_device[key] = value.cuda().to(torch.bfloat16) if value.dtype == torch.float else value.cuda()
#             else:
#                 batch_on_device[key] = value
#         with torch.no_grad():
#             loss = model(**batch_on_device).loss
#         losses.append(loss)
#         print(f'losses: {losses}')
    
# import ipdb; ipdb.set_trace()

# image路径是否正确
for i in tqdm(range(len(train_dataset))):
    meta = train_dataset.list_data_meta[i]
    if isinstance(meta["image"], list):
        image_pathes = [os.path.join(meta["data_path"], img_name) for img_name in meta["image"]]
    else:
        image_pathes = [os.path.join(meta["data_path"], meta["image"])]
    for image_path in image_pathes:
        if not os.path.exists(image_path):
            print(f'image not exist: {image_path}')
            import ipdb; ipdb.set_trace()
