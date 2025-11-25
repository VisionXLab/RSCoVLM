import os
import json

def load_json_or_jsonl(data_root, filename=None):
    data_list = []
    if filename: 
        filename = os.path.join(data_root, filename)
    else:
        filename = data_root
    ext = os.path.splitext(filename)[-1]

    if ext == '.jsonl':
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data_list.append(json.loads(line))
    elif ext == ".json":
        data_list = json.load(open(filename, "r"))
    else:
        raise ValueError(f"Unsupported file type: {ext} (only .json and .jsonl are supported)")
    return data_list

def format_cap(file_name, input_root, output_root):
    breakpoint()
    entry = load_json_or_jsonl(input_root, file_name)
    new_entry = []
    for idx, item in enumerate(entry):
        new_entry.append({
            "image_id": f"{item['image_id'].rsplit('.')[0]}.png",
            "question": "Describe the image",
            "ground_truth": item["description"], 
            "type": "caption",
            "question_id": idx, 
            })

    with open(os.path.join(output_root, file_name), "w", encoding="utf-8") as f:
        for ent in new_entry:
            f.write(json.dumps(ent, ensure_ascii=False) + '\n')
    return

def format_reg_cap(file_name, input_root, output_root):
    entry = load_json_or_jsonl(input_root, file_name)
    new_entry = []
    for idx, item in enumerate(entry):
        item["image_id"] = f"{item['image_id']}.png"
        new_entry.append(item)

    with open(os.path.join(output_root, file_name), "w", encoding="utf-8") as f:
        for ent in new_entry:
            f.write(json.dumps(ent, ensure_ascii=False) + '\n')
    return

def format_cls(file_name, input_root, output_root):
    breakpoint()
    entry = load_json_or_jsonl(input_root, file_name)
    new_entry = []
    for idx, item in enumerate(entry):
        image_id = item["image"]
        if "aid" in file_name.lower():
            image_id = image_id.rsplit(".")[0] + ".jpg"
        gt = "medium residential" if item["ground_truth"] == "MediumResidentialgt" else item["ground_truth"]
        new_entry.append({
            "image_id": image_id,
            "classes": item["text"].rsplit("Classes:")[1].rsplit("\n")[0].strip().lower(),
            "ground_truth": gt.lower(), 
            "type": "cls",
            "question_id": idx, 
            })

    with open(os.path.join(output_root, file_name), "w", encoding="utf-8") as f:
        for ent in new_entry:
            f.write(json.dumps(ent, ensure_ascii=False) + '\n')
    return

def format_vqa(file_name, input_root, output_root, answer_file):
    entry = load_json_or_jsonl(input_root, file_name)
    ans = load_json_or_jsonl(answer_file)
    breakpoint()
    new_entry = []
    for idx, item in enumerate(entry):
        answer = ans["answers"][item["question_id"]]["answer"]
        new_entry.append({
            "image_id": item["image"],
            "question": item["text"].rsplit("\n")[0].strip().lower(),
            "ground_truth": answer.lower(), 
            "category": item["category"],
            "type": "vqa",
            "question_id": item["question_id"], 
            })

    with open(os.path.join(output_root, file_name), "w", encoding="utf-8") as f:
        for ent in new_entry:
            f.write(json.dumps(ent, ensure_ascii=False) + '\n')
    return

def format_ref(file_name, input_root, output_root):
    entry = load_json_or_jsonl(input_root, file_name)
    new_entry = []
    for idx, item in enumerate(entry):
        item["image_id"] = f"{item['image_id'].rsplit('.')[0]}.png"
        new_entry.append(item)

    with open(os.path.join(output_root, file_name), "w", encoding="utf-8") as f:
        for ent in new_entry:
            f.write(json.dumps(ent, ensure_ascii=False) + '\n')
    return

if __name__ == "__main__":
    data_root = "./playground/data/GeoChat"
    benchmark = "geochat_referring" # optional
    output_path = os.path.join(data_root, "converted")
    os.makedirs(output_path, exist_ok=True)

    dataset_name = benchmark.split("_", 1)[-1]
    benchmark_info = {
        'geochat_aid':{'filename': 'aid.jsonl', 'task': 'cls'}, 
        'geochat_grounding_description': {'filename': 'grounding_description.jsonl', 'task': 'caption'},  #
        'geochat_hrben': {'filename': 'hrben.jsonl', 'task': 'vqa'}, #
        'geochat_lrben': {'filename': 'lrben.jsonl', 'task': 'vqa'}, #
        'geochat_referring': {'filename': 'referring.jsonl', 'task': 'referring'},  #
        'geochat_region_captioning': {'filename': 'region_captioning.jsonl', 'task': 'region_caption'}, 
        'geochat_ucmerced': {'filename': 'UCmerced.jsonl', 'task': 'cls'}
    }[benchmark]

    # refgeo lrben/hrben json lacks answer
    lr_answer_file = "/mnt/petrelfs/liqingyun/share_data/data/rsvqa/RSVQA_LR/all_answers.json"
    hr_answer_file = "/mnt/petrelfs/liqingyun/share_data/data/rsvqa/RSVQA_HR/USGSanswers.json"
    task = benchmark_info["task"]
    if task == "caption":
        format_cap(benchmark_info["filename"], data_root, output_path)
    if task == "region_caption":
        format_reg_cap(benchmark_info["filename"], data_root, output_path)
    if task == "cls":
        format_cls(benchmark_info["filename"], data_root, output_path)
    if task == "vqa":
        if "lrben" in benchmark_info["filename"].lower():
            answer_file = lr_answer_file
        elif "hrben" in benchmark_info["filename"].lower():
            answer_file = hr_answer_file
        format_vqa(benchmark_info["filename"], data_root, output_path, answer_file)
    elif task == "referring":
        format_ref(benchmark_info["filename"], data_root, output_path)

    print(f"Finish, saving path: {os.path.join(output_path, benchmark_info['filename'])}")