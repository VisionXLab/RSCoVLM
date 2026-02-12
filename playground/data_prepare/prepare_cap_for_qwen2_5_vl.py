import json
import os
from PIL import Image
from tqdm import tqdm

def format2metainfo(input_json_path, output_jsonl_path, image_root):
    # Load original JSON data
    with open(input_json_path, 'r') as f:
        original_data = json.load(f)
    
    # Process each item and write to JSONL
    with open(output_jsonl_path, 'w') as f_out:
        
        for idx, item in tqdm(enumerate(original_data), 
                             total=len(original_data),
                             desc=f"Converting {os.path.basename(input_json_path)}"):
            # Get image dimensions
            img_path = os.path.join(image_root, item['image_id'])
            try:
                with Image.open(img_path) as img:
                    width, height = img.size # TODO: long-->w, short-->h?
            except:
                print(f"Warning: Could not get dimensions for {img_path}")
                # width, height = 800, 800  # default values
            # Create new format entry
            for gt in item["ground_truth"]:
                new_entry = {
                    "question_id": idx,  # sequential numbering
                    "image_id": item["image_id"],
                    "ground_truth": gt.lower(),
                    # "question": f"Is this a {item['ground_truth']}?",  # Simple classification question
                    "dataset": item['dataset'],
                    "image_width": width,
                    "image_height": height,
                    "task_type": "cap"
                }
                
                # Write as JSON line
                f_out.write(json.dumps(new_entry) + '\n')

if __name__ == "__main__":
    input_json_list = [
        "playground/data/rs_caption/rs_caption_jsons/converted/cap_nwpu_caption_train.json",
        "playground/data/rs_caption/rs_caption_jsons/converted/cap_rsicd_train.json",
    ]
    image_root_list = [
        "playground/data/NWPU-RESISC45",
        "playground/data/rs_caption/rsicd/RSICD_images"
    ]
    new_metainfo_dir = "playground/data/rs_caption/metainfo_v2"
    os.makedirs(new_metainfo_dir, exist_ok=True)

    metainfo_file_list = [
        "cap_nwpu_caption_train_v2.jsonl",
        "cap_rsicd_train_v2.jsonl"
    ]
    
    for input_json, metainfo_file, image_root in zip(input_json_list, metainfo_file_list, image_root_list):
        output_path = os.path.join(new_metainfo_dir, metainfo_file)
        format2metainfo(input_json, output_path, image_root)
        print(f"Converted {input_json} to {output_path}")