import re
import torch
from argparse import Namespace
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from rscovlm.training.data import make_supervised_data_module

CHECK_LOSS = True
model_path = "./playground/Qwen2.5-VL-3B-Instruct"
datasets = ["llava_onevision_vl"]

data_args = Namespace(datasets=datasets)
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
if CHECK_LOSS:
    model = model.cuda().eval()

modules = make_supervised_data_module(processor, data_args, model)
train_dataset, data_collator = modules["train_dataset"], modules["data_collator"]
print(f'total train data: {len(train_dataset)}')

train_dataloder = DataLoader(train_dataset, batch_size=1, collate_fn=data_collator, num_workers=0, shuffle=False)
train_dataiter = iter(train_dataloder)

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
for _ in range(1000):
    batch = next(train_dataiter)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    try:
        pixel_values = batch['pixel_values']
    except:
        import ipdb; ipdb.set_trace()
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

    if CHECK_LOSS:
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.cuda().to(torch.bfloat16) if value.dtype == torch.float else value.cuda()
            else:
                batch_on_device[key] = value
        with torch.no_grad():
            loss = model(**batch_on_device).loss
        losses.append(loss)
        print(f'losses: {losses}')
