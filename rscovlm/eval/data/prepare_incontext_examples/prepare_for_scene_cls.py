"""
import json
resisc_get_cat = lambda x: x.split("_")[0]
with open("./ppl_resisc45_train.json", "r") as f:
    data = json.load(f)
data = [(k, v, resisc_get_cat(k)) for k, v in data.items()]
data = sorted(data, key=lambda x: x[1], reverse=True)
print([s[0] for s in data[:32]])
samples_unique_cat = []
while len(samples_unique_cat) < 32:
    cat_set = set()
    pop_id = []
    for i, s in enumerate(data):
        if s[2] not in cat_set:
            cat_set.add(s[2])
            samples_unique_cat.append(s)
            pop_id.append(i)
    pop_id.reverse()
    for i in pop_id:
        data.pop(i)
print([s[0] for s in samples_unique_cat[:32]])
"""
import json
from tqdm import tqdm

import torch
import torch.distributed as dist

import datasets
from rscovlm.eval.inferencer import get_inferencer
from rscovlm.utils import init_distributed_device, world_info_from_env, partition_for_rank, gather_dict


def calculate_single_round_conversation_ppl(inferencer, pil_image, question, answer):
    msg_instruction = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question}
        ]},
    ]
    msg_instruction_and_response = msg_instruction + [
        {"role": "assistant", "content": answer},
    ]

    text_instruction = inferencer.processor.apply_chat_template(msg_instruction, tokenize=False, add_generation_prompt=True)
    text_instruction_and_reponse = inferencer.processor.apply_chat_template(msg_instruction_and_response, tokenize=False).rstrip("\n")
    text_response = text_instruction_and_reponse.replace(text_instruction, "")
    assert "".join([text_instruction, text_response]) == text_instruction_and_reponse

    pil_imgs = [pil_image]
    all_text_instruction, all_text_response = [text_instruction], [text_response]

    inputs = inferencer.processor(
        text=all_text_instruction, 
        images=pil_imgs, 
        return_tensors="pt", 
        padding=True,
    )
    assert torch.all(inputs.attention_mask)
    inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

    additional_inputs = inferencer.processor.tokenizer(
        text=all_text_response, 
        return_tensors="pt", 
        padding="longest", 
        padding_side="right",
        truncation=True,
    )
    additional_input_ids = additional_inputs["input_ids"]
    additional_attention_mask = additional_inputs["attention_mask"]
    additional_labels = additional_input_ids.clone()
    additional_labels[~additional_attention_mask.bool()] = -100

    inputs["input_ids"] = torch.cat([inputs["input_ids"], additional_input_ids], dim=1)
    inputs["labels"] = torch.cat([inputs["labels"], additional_labels], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], additional_attention_mask], dim=1)

    inputs.to(inferencer.device)
    with torch.no_grad():
        loss = inferencer.model(**inputs).loss
        ppl = torch.exp(loss).item()
    return ppl
    

def sort_res(ppl_results):
    sorted_res = {}
    for k in sorted(ppl_results.keys()):
        sorted_res[k] = ppl_results[k]
    return sorted_res


def prepare_for_scene_cls_timm(model_path, dataset_name):
    _, rank, world_size, _ = world_info_from_env()
    device = init_distributed_device(dist_backend=None)  # using nccl may raise error for 4090 cluster
    inferencer = get_inferencer(
        model_path, 
        device=device, 
        torch_dtype="bf16", 
        attn_implementation="flash_attention_2", 
        lazy_loading=False,
    )

    get_prompt = lambda classes: (
        "Classify the image within one of the given classes:"
        + ",".join(classes) + "." 
        # + " Answer the question using a single word or a short phrase."
    )
    data = datasets.load_dataset(dataset_name, split='train')  # TODO: do we need val?
    classes = data.features['label'].names

    def map_fn(x):
        return {
            "pil_image": x["image"],
            "image_id": x["image_id"],
            "ground_truth": classes[x["label"]],
            "question": get_prompt(classes)
        }
    
    data = list(data.map(map_fn))
    data = partition_for_rank(data, rank, world_size)

    ppl_results = {}
    for x in tqdm(data, disable=(rank!=0)):
        ppl_results[x["image_id"]] = calculate_single_round_conversation_ppl(
            inferencer, x["pil_image"], x["question"], x["ground_truth"])
    
    if world_size > 1:
        ppl_results = gather_dict(ppl_results, rank, world_size)

    if rank == 0:
        with open(f"ppl_{dataset_name.split('/')[-1]}_train.json", "w") as f:
            json.dump(sort_res(ppl_results), f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    model_path = "/mnt/hwfile/share_data/liqingyun/Qwen2.5-VL-3B-Instruct"
    prepare_for_scene_cls_timm(model_path, "timm/resisc45")
    # prepare_for_scene_cls_timm(model_path, "timm/eurosat-rgb")
