import re
import os
import torch
import random
from tqdm import tqdm
from argparse import Namespace
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from rscovlm.training.data import make_supervised_data_module

CHECK_LOSS = False

model_path = "./playground/Qwen2.5-VL-3B-Instruct"
image_folder = "./playground/data/refGeo/"
data_args = Namespace(
    datasets=['refgeo_poly'], # 'refgeo_hbb', 'refgeo_obb'
    min_pixels=256 * 28 * 28, 
    max_pixels=1296 * 28 * 28,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda() if CHECK_LOSS else None
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
import ipdb; ipdb.set_trace()
modules = make_supervised_data_module(processor, data_args, model)
train_dataset, data_collator = modules["train_dataset"], modules["data_collator"]
print(f'all train data: {len(train_dataset)}')

# 检查image是否存在：
for i in tqdm(range(len(train_dataset))):
    metainfo_id = train_dataset.todo_list[i][0]
    image_path = train_dataset.list_data_dict[metainfo_id]['image_file_path']
    if not os.path.exists(image_path):
        print(f'image not exist: {image_path}')
        import ipdb; ipdb.set_trace()
# 获取一个样本，debug数据集的get item
# import ipdb; ipdb.set_trace()
example = train_dataset[3]
breakpoint()
for meta in random.sample(train_dataset.list_data_dict, 50):
    train_dataset.add_resized_size_to_metainfo(meta)
    w, h, wr, hr = meta['image_width'], meta['image_height'], meta['resized_width'], meta['resized_height']
    print(f"image size: {w}x{h} -> {wr}x{hr}")

from torch.utils.data import DataLoader
train_dataloder = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator, num_workers=0, shuffle=True)
train_dataiter = iter(train_dataloder)

# 怎么拼成一个batch，debug数据集的collate_fn/datacollator
example_batch = next(train_dataiter)

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
for _ in range(10):
    batch = next(train_dataiter)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    pixel_values = batch['pixel_values']
    image_grid_thw = batch['image_grid_thw']
    labels = batch['labels']

    print("input_ids_decoded")
    input_ids_decoded = processor.tokenizer.batch_decode(input_ids)
    print(simplify_output(input_ids_decoded)); print()

    print("labels_decoded")
    converted_labels = []
    for label in labels:
        converted_label = label.clone()
        converted_label[converted_label == -100] = 0
        converted_labels.append(converted_label)
    converted_labels = [label.tolist() for label in converted_labels]
    labels_decoded = processor.tokenizer.batch_decode(converted_labels)
    print(simplify_output(labels_decoded)); print()

    print("pure_labels_decoded")
    pure_labels = [label[label != -100].tolist() for label in labels]
    pure_labels_decoded = processor.tokenizer.batch_decode(pure_labels)
    print(simplify_output(pure_labels_decoded)); print()

    print("length wo padding")
    print(attention_mask.sum(dim=1)); print()

    if CHECK_LOSS:
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.cuda().to(torch.bfloat16) if value.dtype == torch.float else value.cuda()
            else:
                batch_on_device[key] = value
        loss = model(**batch_on_device).loss
        losses.append(loss)
    
import ipdb; ipdb.set_trace()
