import os
import json
from tqdm import tqdm
from PIL import Image


def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data


def save_wh_maskexists_to_metainfo_v2():
    metainfo_file_list = [
        # "avvg_detection_train.jsonl",
        # "rrsisd_train.jsonl",
        "avvg_train.jsonl",
        "dior_rsvg_train.jsonl",
        "geochat_train.jsonl",
        "rsvg_train.jsonl",
        "vrsbench_train.jsonl"
    ]
    metainfo_dir = "playground/data/refGeo/metainfo"
    new_metainfo_dir = "playground/data/refGeo/metainfo_v2"
    image_dir = "playground/data/refGeo/images"
    mask_dir = "playground/data/refGeo/masks"

    for metainfo_file in metainfo_file_list:
        meta_info = load_jsonl(os.path.join(metainfo_dir, metainfo_file))
        for meta in tqdm(meta_info, desc=f"Processing {metainfo_file}"):
            if "geoground" in metainfo_file:
                image_surfix = f"geoground/{meta['image_id']}"
                mask_surfix = f"geoground/{meta['image_id'].split('.')[0] + '_' + str(meta['question_id']) + '.png'}"
                import ipdb; ipdb.set_trace()
            else:
                image_surfix = f"{meta['dataset']}/{meta['image_id']}"
                mask_surfix = f"{meta['dataset']}/{meta['image_id'].split('.')[0] + '_' + str(meta['question_id']) + '.png'}"
            image_file_path = f"{image_dir}/{image_surfix}"
            mask_path = f"{mask_dir}/{mask_surfix}"

            pil_image = Image.open(image_file_path)
            image_width, image_height = pil_image.size
            meta["image_width"] = image_width
            meta["image_height"] = image_height

            mask_exists = os.path.exists(mask_path)
            meta["mask_exists"] = mask_exists
            if mask_exists:
                mask = Image.open(mask_path)
                mask_width, mask_height = mask.size
                assert image_width == mask_width and image_height == mask_height, f"Image and mask sizes do not match for {image_surfix}"

        metainfo_v2_file = metainfo_file.replace(".jsonl", "_v2.jsonl")
        new_metainfo_file_path = os.path.join(new_metainfo_dir, metainfo_v2_file)
        os.makedirs(new_metainfo_dir, exist_ok=True)
        with open(new_metainfo_file_path, 'w') as new_metainfo_file:
            for meta in meta_info:
                new_metainfo_file.write(json.dumps(meta, ensure_ascii=False) + '\n')


def check_new_metainfo():
    metainfo_file_list = [
        # "avvg_detection_train_v2.jsonl",
        # "rrsisd_train_v2.jsonl",
        "avvg_train_v2.jsonl",
        "dior_rsvg_train_v2.jsonl",
        "geochat_train_v2.jsonl",
        "rsvg_train_v2.jsonl",
        "vrsbench_train_v2.jsonl"
    ]
    new_metainfo_dir = "playground/data/refGeo/metainfo_v2"

    for metainfo_file in metainfo_file_list:
        new_metainfo_file_path = os.path.join(new_metainfo_dir, metainfo_file)
        meta_info = load_jsonl(new_metainfo_file_path)

        num_mask_exists = sum(meta["mask_exists"] for meta in meta_info)
        print(f"{metainfo_file} - Number of images with masks: {num_mask_exists} - Total: {len(meta_info)} - Ratio: {num_mask_exists / len(meta_info):.2%}")


if __name__ == "__main__":
    check_new_metainfo()
