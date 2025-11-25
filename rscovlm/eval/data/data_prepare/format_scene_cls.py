import os
import json
import random
from collections import defaultdict

def convert_to_classification_resisc(output_json, reference_json, dataset_name):
    output_path = os.path.basename(output_json)
    os.makedirs(output_path, exist_ok=True)
    with open(reference_json, 'r') as f:
        data = json.load(f)

    all_classes = list(sorted({item['image_id'].split('/')[0] for item in data}))
    class_str = ','.join(all_classes) + '.'

    new_data = []
    for item in data:
        
        new_item = {
            'image_id': item['image_id'],
            "ground_truth": item['image_id'].rsplit('/')[0],
            'dataset': dataset_name,
            'type': 'cls',
            'split': item['split'],
            'classes': class_str
        }
        new_data.append(new_item)
    
    # 保存转换后的文件
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Conversion complete. Saved to {output_json}")

def convert_to_classification_aid(output_json, data_root, benchmark_name, ratio):
    output_path = os.path.dirname(output_json)
    os.makedirs(output_path, exist_ok=True)
    class_images = defaultdict(list) # class: [imgs]
    classes = set()

    for class_name in os.listdir(data_root):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # 收集该类别所有图像
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                class_images[class_name].append(img_name)
        classes.add(class_name)
    classes = list(sorted(classes))
    classes_str = ','.join(classes) + '.'
    train_data = []
    test_data = []
    
    for class_name, img_list in class_images.items():
        random.shuffle(img_list)
        split_idx = int(len(img_list) * ratio)
        
        # train
        for img_name in img_list[:split_idx]:
            train_data.append({
                'image_id': f"{class_name}/{img_name}",
                'ground_truth': class_name,
                'dataset': benchmark_name,
                'type': 'cls',
                'split': 'train',
                'classes': classes_str
            })
        
        # test
        for img_name in img_list[split_idx:]:
            test_data.append({
                'image_id': f"{class_name}/{img_name}",
                'ground_truth': class_name,
                'dataset': benchmark_name,
                'type': 'cls',
                'split': 'test',
                'classes': classes_str
            })

    with open(os.path.join(output_path, f"{benchmark_name}_train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(output_path, f"{benchmark_name}_test.json"), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f'data saved to {output_path}')
    print(f"train data num: {len(train_data)}")
    print(f"test data num: {len(test_data)}")


if __name__ == "__main__":

    output_json = './playground/data/scene_cls/cls_resisc_val.json'
    benchmark_name = 'cls_aid' # 'cls_nwpu_resisc45'
    
    if benchmark_name == 'cls_nwpu_resisc45':
        reference_json = './playground/data/rs_caption/rs_caption_jsons/converted/cap_nwpu_caption_val.json'
        convert_to_classification_resisc(output_json, reference_json, benchmark_name)
    elif benchmark_name == 'cls_aid':
        '''
        For the RSSCN7 dadaset and our AID dataset, we fix the ratio of the number of training set to
        be 20% and 50% respectively and the left for testing, 
        '''
        ratio = 0.2
        data_root = './playground/data/AID'
        convert_to_classification_aid(output_json, data_root, benchmark_name, ratio)