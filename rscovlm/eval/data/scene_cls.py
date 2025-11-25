import gc
import os
import json
import logging
import datasets
from tqdm import tqdm

import torch.distributed as dist
from torchvision.datasets.folder import find_classes, make_dataset, IMG_EXTENSIONS

from .data import get_incontext_msg

logger = logging.getLogger(__name__)


def prepare_scene_cls(data_root, from_timm=False):
    get_prompt = lambda classes: (
        "Classify the image within one of the given classes:"
        + ",".join(classes) + "." 
        # + " Answer the question using a single word or a short phrase."  # TODO: whether to add this?
    )
    if from_timm:
        data = datasets.load_dataset(data_root, split='test')
        classes = data.features['label'].names

        def map_fn(x):
            return {
                "pil_image": x["image"],
                "image_id": x["image_id"],
                "ground_truth": classes[x["label"]],
                "question": get_prompt(classes)
            }
        
        data = list(data.map(map_fn))
    else:
        classes, class_to_idx = find_classes(data_root)
        data = make_dataset(data_root, class_to_idx=class_to_idx, extensions=IMG_EXTENSIONS)

        def map_fn(image_path, class_idx):
            return {
                "image_path": image_path,
                "ground_truth": classes[class_idx],
                "question": get_prompt(classes)
            }

        data = [map_fn(image_path, class_idx) for image_path, class_idx in data]
    logger.info(get_prompt(classes))

    for idx, x in enumerate(data):
        if 'image_path' in x:
            image_path = x["image_path"]
            image = f"file://{image_path}"
        elif 'pil_image' in x:
            image = x["pil_image"]

        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": x['question']}
            ]}
        ]

        x["message"] = message
        x["idx"] = idx
    return data


def prepare_scene_cls_local(image_root, benchmark, icl_shot, data_json_root, ppl_json_path, unique):
    incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path, unique)
    test_data = json.load(open(os.path.join(data_json_root, f'{benchmark}_test.json'), "r"))
    for idx, x in enumerate(test_data):
        classes_str = x["classes"]
        image_path = os.path.join(image_root, x['image_id'])
        question = "Classify the image within one of the given classes:" + classes_str
        message = incontext_msg + [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]}
        ]
        x["message"] = message
        x["idx"] = idx
    return test_data


# support max 32 incontext samples
RESISC_HARD_TRAINING_SAMPLES = [
    'ship_125', 'ship_237', 'ship_090', 'church_432', 'ship_626', 
    'ship_683', 'intersection_048', 'intersection_190', 'intersection_003', 
    'palace_533', 'ship_091', 'lake_669', 'palace_675', 'overpass_449', 
    'palace_032', 'lake_363', 'church_513', 'intersection_425', 'ship_296', 
    'intersection_593', 'palace_536', 'palace_455', 'palace_686', 'palace_578', 
    'palace_325', 'ship_277', 'intersection_500', 'palace_534', 'lake_599', 
    'palace_282', 'palace_118', 'palace_443'
]
RESISC_HARD_TRAINING_SAMPLES_UNIQUE_CLASS = [
    'ship_125', 'church_432', 'intersection_048', 'palace_533', 'lake_669', 
    'overpass_449', 'river_141', 'cloud_360', 'chaparral_132', 'harbor_514', 
    'freeway_091', 'terrace_071', 'meadow_286', 'railway_075', 
    'thermal_power_station_489', 'commercial_area_635', 'roundabout_271', 
    'desert_586', 'wetland_465', 'island_238', 'mountain_365', 'runway_287', 
    'snowberg_244', 'tennis_court_266', 'beach_628', 'airplane_613', 
    'storage_tank_076', 'sparse_residential_091', 'mobile_home_park_029', 
    'ground_track_field_442', 'industrial_area_633', 'dense_residential_530'
]
EUROSAT_HARD_TRAINING_SAMPLES = [
    'River_1006', 'River_610', 'River_1730', 'River_1179', 'River_544', 
    'River_464', 'River_1642', 'Industrial_1676', 'River_1244', 'River_1897', 
    'Industrial_471', 'River_1757', 'Industrial_562', 'River_838', 'River_1118', 
    'Industrial_1112', 'Industrial_1653', 'River_1292', 'Industrial_2129', 
    'River_1740', 'Industrial_1816', 'River_1714', 'Industrial_53', 
    'Industrial_283', 'Industrial_1380', 'River_2165', 'River_709', 
    'Industrial_2360', 'River_1064', 'Industrial_333', 'River_1516', 
    'Industrial_2132',
]
EUROSAT_HARD_TRAINING_SAMPLES_UNIQUE_CLASS = [
    'River_1006', 'Industrial_1676', 'SeaLake_512', 'Pasture_1374', 
    'Residential_2913', 'AnnualCrop_1188', 'Forest_2632', 'PermanentCrop_1748', 
    'Highway_869', 'HerbaceousVegetation_1921', 'River_610', 'Industrial_471', 
    'SeaLake_15', 'Pasture_1102', 'Residential_869', 'AnnualCrop_1744', 
    'PermanentCrop_1878', 'Forest_1777', 'Highway_1485', 
    'HerbaceousVegetation_1466', 'River_1730', 'Industrial_562', 'SeaLake_1871', 
    'Pasture_1059', 'Residential_1937', 'AnnualCrop_2406', 'PermanentCrop_2385', 
    'Forest_909', 'Highway_1737', 'HerbaceousVegetation_25', 'River_1179', 
    'Industrial_1112',
]


def prepare_scene_cls_timm_incontext(data_root, num_incontext_samples=1, unique_class_mode=False):  # setting 3
    if "resisc" in data_root:
        offline_incontext_samples_ids = RESISC_HARD_TRAINING_SAMPLES_UNIQUE_CLASS \
            if unique_class_mode else RESISC_HARD_TRAINING_SAMPLES
    elif "eurosat" in data_root:
        offline_incontext_samples_ids = EUROSAT_HARD_TRAINING_SAMPLES_UNIQUE_CLASS \
            if unique_class_mode else EUROSAT_HARD_TRAINING_SAMPLES
    else:
        raise ValueError(f"Unsupported data_root: {data_root}")
    
    if num_incontext_samples > len(offline_incontext_samples_ids):
        raise ValueError("num_incontext_samples should be less than or equal to the number of incontext samples")
    offline_incontext_samples_ids = offline_incontext_samples_ids[:num_incontext_samples]

    get_prompt = lambda classes: (
        "Classify the image within one of the given classes:"
        + ",".join(classes) + "." 
        # + " Answer the question using a single word or a short phrase."
    )

    logger.info(f"Loading test set from huggingface hub {data_root}...")
    data = datasets.load_dataset(data_root, split='test')
    classes = data.features['label'].names

    def map_fn(x):
        return {
            "pil_image": x["image"],
            "image_id": x["image_id"],
            "ground_truth": classes[x["label"]],
            "question": get_prompt(classes)
        }

    logger.info(f"Mapping test samples...")
    data = data.map(map_fn)
    logger.info(get_prompt(classes))

    logger.info(f"Loading train set from huggingface hub {data_root}...")
    incontext_data = datasets.load_dataset(data_root, split='train')

    logger.info(f"Preparing incontext message...")
    incontext_message = []
    incontext_data = {item["image_id"]: map_fn(item) for item in incontext_data if item["image_id"] in offline_incontext_samples_ids}
    for i in range(num_incontext_samples):
        incontext_sample = incontext_data[offline_incontext_samples_ids[i]]
        if 'image_path' in incontext_sample:
            image_path = incontext_sample["image_path"]
            image = f"file://{image_path}"
        elif 'pil_image' in incontext_sample:
            image = incontext_sample["pil_image"]
        incontext_message.extend([
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": incontext_sample['question']}
            ]}, 
            {"role": "assistant", "content": incontext_sample['ground_truth']},
        ])
    del incontext_data
    gc.collect()

    def data_generator():
        rank = dist.get_rank() if dist.is_initialized() else 0
        for idx, x in enumerate(tqdm(data, desc='Generating data (maybe along with the model inferencing)', disable=(rank != 0))):
            if 'image_path' in x:
                image_path = x["image_path"]
                image = f"file://{image_path}"
            elif 'pil_image' in x:
                image = x["pil_image"]

            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": x['question']}
                ]}
            ]

            x["message"] = incontext_message + message
            x["idx"] = idx
            yield x
    return data_generator()
