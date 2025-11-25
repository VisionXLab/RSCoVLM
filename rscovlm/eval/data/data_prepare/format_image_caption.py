import argparse
import json
import logging
import os
from tqdm import tqdm
from typing import List, Dict, Tuple

def save_split(data: List[Dict], filename: str, output_dir: str):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} items to {path}")

def do_nwpu_json(original_data, dataset_name):
    '''
    Split dataset and re-prefix image paths.
    Now organizes data by image (one entry per image with multiple captions)
    '''
    # initialize
    train_data = []
    val_data = []
    test_data = []
    image_id_counter = 0  
    # for each category
    for category, items in tqdm(original_data.items(), desc="Processing categories"):
        for item in items:  
            raw_fields = [k for k in item.keys() if k.startswith('raw')]
            ground_truths = [item[raw_field].strip() for raw_field in raw_fields]
            new_entry = {
                "image_id": f"{category}/{item['filename']}",
                "question": "Describe the image",
                "dataset": dataset_name,
                "type": "caption",
                "ground_truth": ground_truths,  
                "question_id": image_id_counter, 
                "split": item["split"]
            }
            image_id_counter += 1
            # split the json
            if item["split"] == "train":
                train_data.append(new_entry)
            elif item["split"] == "val":
                val_data.append(new_entry)
            elif item["split"] == "test":
                test_data.append(new_entry)
            else:
                error_msg = f"Unknown split: {item['split']} for image {item['filename']}"
                raise AssertionError(error_msg)
    return train_data, val_data, test_data

def do_json(original_data, dataset_name):
    '''
    Split dataset and re-prefix image paths.
    Now organizes data by image (one entry per image with all captions)
    '''
    train_data = []
    val_data = []
    test_data = []
    image_id_counter = 0  # Tracks unique image IDs
    for image in tqdm(original_data["images"], desc='Processing Images'):
        ground_truths = [sentence["raw"].strip() for sentence in image["sentences"]]
        new_entry = {
            "image_id": image["filename"],          
            "question": "Describe the image",       
            "dataset": dataset_name,               
            "type": "caption",                   
            "ground_truth": ground_truths,       
            "question_id": image_id_counter,   
            "split": image["split"] 
        }
        image_id_counter += 1
        if image["split"] == "train":
            train_data.append(new_entry)
        elif image["split"] == "val":
            val_data.append(new_entry)
        elif image["split"] == "test":
            test_data.append(new_entry)
        else:
            error_msg = f"Unknown split: {image['split']} for image {image['filename']}"
            raise AssertionError(error_msg)
    return train_data, val_data, test_data

def json_normalization(benchmark, json_path):
    json_filename = benchmark + '.json' # should be named as benchmark.json
    print(f'Formatting {benchmark}.json ...')
    with open(os.path.join(json_path, json_filename), 'r') as f:
        original_data = json.load(f)
    if benchmark == 'cap_nwpu_caption': # NWPU_caption has a different organization format
        train_data, val_data, test_data = do_nwpu_json(original_data, benchmark)
    else:
        train_data, val_data, test_data = do_json(original_data, benchmark)

    output_dir=os.path.join(json_path, 'converted')
    os.makedirs(output_dir, exist_ok=True)
    if train_data:
        save_split(train_data, f'{benchmark}_train.json', output_dir)
    if val_data:
        save_split(val_data, f'{benchmark}_val.json', output_dir)
    if test_data:
        save_split(test_data, f'{benchmark}_test.json', output_dir)
    return output_dir

if __name__ == "__main__":
    benchmarks = ['cap_nwpu_caption', 'cap_rsicd', 'cap_rsitmd', 'cap_sydney_caption', 'cap_ucm_caption']
    rs_caption_json_root="./playground/data/rs_caption/rs_caption_jsons"
    for benchmark in benchmarks:
        converted_json_dir = json_normalization(benchmark, rs_caption_json_root)