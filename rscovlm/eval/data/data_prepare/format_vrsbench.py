import os
import json
import re
from typing import List, Dict, Tuple
from tqdm import tqdm

def detect_benchmark_type(question: str, gt: str) -> str:
    """Automatically detect benchmark type based on question keywords."""
    question_lower = question.lower()
    caption_keywords = {'describe', 'description'}
    referring_pattern = r'^\{(?:<\d+>)+\}$'
    if any(keyword in question_lower for keyword in caption_keywords):
        return 'caption'
    elif bool(re.fullmatch(referring_pattern, gt)):
        return 'referring'
    else:
        return 'vqa'

def process_caption_item(item: Dict, idx: int) -> Dict:
    """Process an item for caption benchmark."""
    question = item["conversations"][0]["value"].replace("<image>\n", "").strip()
    return {
        "image_id": item["image"],
        "ground_truth": [item["conversations"][1]["value"]],
        "question": question,
        "dataset": "RSBench",
        "question_id": idx,
        "type": "caption"
    }

def process_referring_item(item: Dict, idx: int) -> Dict:
    """Process an item for referring benchmark."""
    question = item["conversations"][0]["value"].replace("<image>\n", "").strip()
    # clean_question = question.replace('<image>', '').replace('<p>', '').replace('</p>', '').strip().split('"value": "')[-1]
    raw_gt = [item["conversations"][1]["value"]][0]
    clean_gt = [int(num) for num in re.findall(r'<(\d+)>', raw_gt)]
    pattern = r'<p>(.*?)<\/p>'
    normal_caption = re.findall(pattern, question)[0]
    if normal_caption.lower().startswith('the '):
        normal_caption = normal_caption[4:].strip()
    return {
        "image_id": item["image"],
        "normal_caption": normal_caption,
        "solution": clean_gt,
        "normalized_solution": clean_gt,
        "question": question,
        "dataset": "RSBench",
        "question_id": idx,
        "type": "referring"
    }

def process_vqa_item(item: Dict, idx: int) -> Dict:
    """Process an item for VQA benchmark."""
    question = item["conversations"][0]["value"].replace("<image>\n", "").strip()
    return {
        "image_id": item["image"],
        "ground_truth": [item["conversations"][1]["value"]],
        "question": question,
        "dataset": "RSBench",
        "question_id": idx,
        "type": "vqa"
    }

def save_split(data: List[Dict], filename: str, save_path: str):
    """Save data to JSON file."""
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(data, f, indent=2)

def get_item_key(item: Dict, benchmark_type: str) -> Tuple:
    """Generate a unique key for deduplication based on benchmark type."""
    question = item["conversations"][0]["value"].replace("<image>\n", "").strip().lower()
    image_id = item["image"]
    if benchmark_type == 'referring':
        raw_gt = item["conversations"][1]["value"]
        gt_key = tuple(int(num) for num in re.findall(r'<(\d+)>', raw_gt))
    else:
        gt_key = item["conversations"][1]["value"].lower()
    return (image_id, question, gt_key)

def process_rsbench_train_data(train_data: List[Dict], data_root: str, benchmarks: Dict) -> Dict[str, List[Dict]]:
    """
    Process RSBench train data and automatically split into different benchmark types.
    Returns a dictionary mapping benchmark types to their processed data.
    """
    benchmark_data = {
        'caption': [],
        'referring': [],
        'vqa': []
    }
    seen_items = {
        'caption': set(),
        'referring': set(),
        'vqa': set()
    }
    
    for idx, item in tqdm(enumerate(train_data)):
        question = item["conversations"][0]["value"].replace("<image>\n", "").strip()
        gt = item["conversations"][1]["value"].strip() # to match referring
        benchmark_type = detect_benchmark_type(question, gt)
        key = get_item_key(item, benchmark_type)
        if key not in seen_items[benchmark_type]:
            seen_items[benchmark_type].add(key)
            if benchmark_type == 'caption':
                processed_item = process_caption_item(item, idx)
                benchmark_data['caption'].append(processed_item)
            elif benchmark_type == 'referring':
                processed_item = process_referring_item(item, idx)
                benchmark_data['referring'].append(processed_item)
            else:  # vqa
                processed_item = process_vqa_item(item, idx)
                benchmark_data['vqa'].append(processed_item)
    
    # Save each benchmark type separately
    norm_json_save_path = os.path.join(data_root, 'converted')
    os.makedirs(norm_json_save_path, exist_ok=True)
    
    for benchmark_type, data in benchmark_data.items():
        benchmark = benchmarks[benchmark_type]
        save_split(data, f'{benchmark}_train.json', norm_json_save_path)
    
    return norm_json_save_path
def process_rsbench_referring_test_data(test_referring_data, data_root, benchmark):
    norm_json_save_path = os.path.join(data_root, 'converted')
    os.makedirs(norm_json_save_path, exist_ok=True)
    entries = []
    for idx, item in enumerate(tqdm(test_referring_data)):
        """Process an item for referring benchmark."""
        normal_caption = item["question"].replace("<image>\n", "").strip() # actually it is 'normal_caption', not a question format.
        if normal_caption.lower().startswith('the '):
            normal_caption = normal_caption[4:].rstrip(".")
        # clean_question = question.replace('<image>', '').replace('<p>', '').replace('</p>', '').strip().split('"value": "')[-1]
        raw_gt = item["ground_truth"]
        clean_gt = [int(num) for num in re.findall(r'<(\d+)>', raw_gt)]    
        new_entry = {
            "image_id": item["image_id"],
            "normal_caption": normal_caption,
            "solution": clean_gt,
            "normalized_solution": clean_gt,
            "dataset": "RSBench",
            "question_id": idx,
            "type": "referring"
        }
        entries.append(new_entry)
    save_split(entries, f'{benchmark}_test.json', norm_json_save_path)
    return norm_json_save_path

if __name__ == "__main__":
    train_json_path = './playground/data/VRSBench/RSBench_train.json'
    test_referring_json = './playground/data/VRSBench/RSBench_EVAL_referring.json'
    benchmarks = {'caption': 'vrsbench_caption', 'referring': 'vrsbench_referring', 'vqa': 'vrsbench_vqa'}
    data_root = os.path.dirname(train_json_path)
    
    # Load the training data
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    # Process and save the data
    save_path = process_rsbench_train_data(train_data, data_root, benchmarks)
    print(f"Processed data saved to: {save_path}")
    test_referring_data = json.load(open(test_referring_json, "r"))
    test_save_path = process_rsbench_referring_test_data(test_referring_data, data_root, 'vrsbench_referring')
    print(f"Processed data saved to: {save_path}")