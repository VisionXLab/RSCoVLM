import torch
from swift.llm import TrainArguments
from swift.llm.train import SwiftSft
from swift.llm.dataset import PackingDataset

kwargs = dict(
    model="./playground/Qwen2.5-VL-3B-Instruct",
    train_type="lora",
    dataset="./playground/data/refGeo/sft/swift_hbb_dior_rsvg_512_random_samples.json",
    split_dataset_ratio=0,  # valset
    torch_dtype="bfloat16",
    attn_impl="flash_attn",
    packing=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    lora_rank=8,
    lora_alpha=32,
    target_modules="model.layers.35.mlp",
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    max_length=4096,
    output_dir="tmp",
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    dataset_num_proc=16,
)

sft = SwiftSft(TrainArguments(**kwargs))
args = sft.args

# test get dataset (not preprocessed)
train_dataset, val_dataset = sft._get_dataset()
assert val_dataset is None
train_dataset = train_dataset.select(range(512))  # has not been preprocessed
example_wo_preprocessed = train_dataset[0]

# test preprocess an example
example = sft.template.encode(train_dataset[0])

# pack trainset (in sft._encode_dataset method)
packed_train_dataset = PackingDataset(
    sft.template, 
    train_dataset, 
    num_workers=1, 
    strict=args.strict
)

# get a packed example (has been preprocessed)
packed_example = packed_train_dataset[0]

print(f"packed_example.keys(): {packed_example.keys()}")
packed_input_ids = packed_example['input_ids']
print(f"packed_input_ids: {torch.as_tensor(packed_input_ids).shape}")
packed_position_ids = packed_example['position_ids']
print(f"position_ids.shape: {torch.as_tensor(packed_position_ids).shape}")
packed_real_position_ids = packed_example['real_position_ids']
print(f"real_position_ids.shape: {torch.as_tensor(packed_real_position_ids).shape}")

# get a batch of packed examples with collator
data_collator = sft._get_data_collator()
batch_example = data_collator(packed_train_dataset[:4])
print(f"batch_example.keys(): {batch_example.keys()}")

input_ids = batch_example['input_ids']
print(f"input_ids.shape: {input_ids.shape}")
labels = batch_example['labels']
print(f"labels.shape: {labels.shape}")
position_ids = batch_example['position_ids']
print(f"position_ids.shape: {position_ids.shape}")
pixel_values = batch_example['pixel_values']
print(f"pixel_values.shape: {pixel_values.shape}")
image_grid_thw = batch_example['image_grid_thw']
print(f"image_grid_thw.shape: {image_grid_thw.shape}")
real_position_ids = batch_example['real_position_ids']
print(f"real_position_ids.shape: {real_position_ids.shape}")

print(f'real_position_ids: ')
pos = real_position_ids[:, 0]  # shape: [3, L]
text_mask = (pos[0] == pos[1]) & (pos[1] == pos[2])
image_mask = ~text_mask
print("text_length:", text_mask.sum().item())
print("image token numbers:", image_mask.sum().item())
print("image t values:", pos[0][image_mask].unique())
print("image h values:", pos[1][image_mask].unique())
print("image w values:", pos[2][image_mask].unique())

import re
def simplify_output(text):
    # return text
    if isinstance(text, list):
        return [simplify_output(t) for t in text]
    patterns = [r'<\|image_pad\|>', r'<\|endoftext\|>', r'!']
    for pattern in patterns:
        text = re.sub(rf'({pattern})\1+', lambda m: f"{m.group(1)} * {len(m.group(0)) // len(m.group(1))} ", text)
    return text

from copy import deepcopy
from colorama import init, Fore, Back, Style
init(autoreset=True)
def print_with_highlight(text, keywords: list[str]):
    def _process(_text, keywords):
        # _text = deepcopy(_text)
        # if isinstance(keywords, str):
        #     keywords = [keywords]
        # for keyword in keywords:
        #     _text = _text.replace(keyword, f"{Back.YELLOW}{Fore.BLACK}{keyword}{Style.RESET_ALL}")
        return _text
    if isinstance(text, list):
        for t in text:  
            print(_process(t, keywords))
    else:
        print(text)

print("input_ids_decoded")
input_ids_decoded = sft.template.processor.tokenizer.batch_decode(input_ids)
print_with_highlight(simplify_output(input_ids_decoded), '<im_start>system'); print()

print("labels_decoded")
converted_labels = []
for label in labels:
    converted_label = label.clone()
    converted_label[converted_label == -100] = 0
    converted_labels.append(converted_label)
converted_labels = [label.tolist() for label in converted_labels]
labels_decoded = sft.template.processor.tokenizer.batch_decode(converted_labels)
print_with_highlight(simplify_output(labels_decoded), '<|im_end|>'); print()

print("pure_labels_decoded")
pure_labels = [label[label != -100].tolist() for label in labels]
pure_labels_decoded = sft.template.processor.tokenizer.batch_decode(pure_labels)
print_with_highlight(simplify_output(pure_labels_decoded), '<|im_end|>'); print()

# test runing training
sft.run()

# test streaming
streaming_sft = SwiftSft(TrainArguments(
    model="./playground/Qwen2.5-VL-3B-Instruct",
    train_type="lora",
    dataset="./playground/data/refGeo/sft/swift_hbb_dior_rsvg_512_random_samples.json",
    split_dataset_ratio=0,  # valset
    torch_dtype="bfloat16",
    attn_impl="flash_attn",
    packing=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    lora_rank=8,
    lora_alpha=32,
    target_modules="model.layers.35.mlp",
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    max_length=4096,
    output_dir="tmp",
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    dataset_num_proc=16,
))


from copy import deepcopy
streaming_kwargs = deepcopy(kwargs)
streaming_kwargs["streaming"] = True
streaming_kwargs["max_steps"] = 100
streaming_kwargs["num_train_epochs"] = -1
streaming_kwargs["per_device_train_batch_size"] = 1
streaming_sft = SwiftSft(TrainArguments(**streaming_kwargs))
streaming_sft.run()

import ipdb; ipdb.set_trace()
