import re
import json
import yaml
from typing import Union
from copy import deepcopy

from .synonyms import *


VHM_DATASET = {
    "annotation_path": "./playground/data/VHM_dataset_sft/vhm_sft_mix_qwen2_5_vl_conversations.json",
    "data_path": "./playground/data/VHM_dataset_sft/images/",
}
REFGEO_AVVG_DATASET = {
    "annotation_path": "./playground/data/refGeo/metainfo_v2/avvg_train_v2.jsonl",
    "data_path": "./playground/data/refGeo/",
    "dataset_type": "refgeo",
}
REFGEO_DIORRSVG_DATASET = {
    "annotation_path": "./playground/data/refGeo/metainfo_v2/dior_rsvg_train_v2.jsonl",
    "data_path": "./playground/data/refGeo/",
    "dataset_type": "refgeo"
}
REFGEO_GEOCHAT_DATASET = {
    "annotation_path": "./playground/data/refGeo/metainfo_v2/geochat_train_v2.jsonl",
    "data_path": "./playground/data/refGeo/",
    "dataset_type": "refgeo"
}
REFGEO_RSVG_DATASET = {
    "annotation_path": "./playground/data/refGeo/metainfo_v2/rsvg_train_v2.jsonl",
    "data_path": "./playground/data/refGeo/",
    "dataset_type": "refgeo"
}
REFGEO_VRSBENCH_DATASET = {
    "annotation_path": "./playground/data/refGeo/metainfo_v2/vrsbench_train_v2.jsonl",
    "data_path": "./playground/data/refGeo/",
    "dataset_type": "refgeo"
}
COCO_TRAIN2017_DATASET = {
    "annotation_path": "./playground/data/coco/annotations/instances_train2017.json",
    "data_path": "./playground/data/coco/train2017/",
    "dataset_type": "dense_det",
}
LRSVQA_TRAIN_VISUAL_COT_OPENENDED_DATASET = [
    {"annotation_path": "./playground/data/LRS_VQA/annotations/train/7_8_GPT4V_LS-VQA_DOTA2-Bridge-STAR_with_boxx_info_159k_imgpath-processed.json"},
    {"annotation_path": "./playground/data/LRS_VQA/annotations/train/7_8_template_DOTA-Bridge-STAR_with_boxx_info_60k_imgpath-processed.json"},
]
LRSVQA_TRAIN_VISUAL_COT_MCQ_DATASET = [{"annotation_path": "./playground/data/LRS_VQA/annotations/train/7_9_transformed_MCQ_83k_imgpath-processed.json"}]
LRSVQA_TRAIN_VISUAL_COT_DATASET = LRSVQA_TRAIN_VISUAL_COT_OPENENDED_DATASET + LRSVQA_TRAIN_VISUAL_COT_MCQ_DATASET
for item in LRSVQA_TRAIN_VISUAL_COT_DATASET:
    item['data_path'] = "./playground/data/LRS_VQA/images/train/"
    item['dataset_type'] = 'visual_cot'

# dense det
DOTA_TRAINVAL1024_DATASET = {
    "annotation_path": "./playground/data/detection/split_ss_dota/trainval.json",
    "data_path": "./playground/data/detection/split_ss_dota/trainval/images/",
}
DOTA_TRAINVAL512_DATASET = {
    "annotation_path": "./playground/data/detection/split_ss_dota_512/trainval.json",
    "data_path": "./playground/data/detection/split_ss_dota_512/trainval/images/",
}
FAIR1M_TRAIN682_DATASET = {
    "annotation_path": "./playground/data/detection/split_ss_fair1m_1_0/train/trainval_1024_P2Bfmt_dota_rbox.json", 
    "data_path": "./playground/data/detection/split_ss_fair1m_1_0/train/images/",
}
DIOR_TRAINVAL800_DATASET = {
    "annotation_path": "./playground/data/detection/DIOR/Annotations/trainval_rbox_pt_P2Bfmt.json",
    "data_path": "./playground/data/detection/DIOR/JPEGImages/",
}
SRSDD_TRAIN1024_DATASET = {
    "annotation_path": "./playground/data/detection/SRSDD/train.json",
    "data_path": "./playground/data/detection/SRSDD/train/images/",
}
RSAR_TRAINVAL190TO1000_DATASET = {
    "annotation_path": "./playground/data/detection/RSAR/trainval.json",
    "data_path": "./playground/data/detection/RSAR/",
}
BH_POOLS_DATASET = {
    "annotation_path": "./playground/data/detection/BH-DATASET/BH-POOLS/train.json",
    "data_path": "./playground/data/detection/BH-DATASET/BH-POOLS/",
}
BH_WATERTANKS_DATASET = {
    "annotation_path": "./playground/data/detection/BH-DATASET/BH-WATERTANKS/train.json",
    "data_path": "./playground/data/detection/BH-DATASET/BH-WATERTANKS/",
}
HRSC2016_DATASET = {
    "annotation_path": "./playground/data/detection/HRSC2016_dataset/train.json",
    "data_path": "./playground/data/detection/HRSC2016_dataset/HRSC2016/HRSC2016/Train/AllImages/",
}
LLAVA_ONEVISION_VL_DATASET = {
    "data_path": "./playground/data/LLaVA-OneVision-Data/shuffled-multimodal/", 
    "dataset_type": "llava_onevision",
}
LLAVA_ONEVISION_TEXT_DATASET = {
    "data_path": "./playground/data/LLaVA-OneVision-Data/shuffled-pure-text/",
    "dataset_type": "llava_onevision",
}
PROCESSED_GEOCHAT_FROM_TEOCHATLAS_DATASET = {
    "annotation_path": "./playground/data/TEOChatlas/train/instruct_GeoChat_modified.json",
    "data_path": "./playground/data/geochat_data/images/",
    "dataset_type": "teochat",
}
TEOCHATLAS_FMOW_DATASET = {"annotation_path": "./playground/data/TEOChatlas/train/instruct_fMoW.json"}
TEOCHATLAS_QFABRIC_DATASET = {"annotation_path": "./playground/data/TEOChatlas/train/instruct_QFabric.json"}
TEOCHATLAS_S2LOOKING_DATASET = {"annotation_path": "./playground/data/TEOChatlas/train/instruct_S2Looking.json"}
TEOCHATLAS_XBD_DATASET = {"annotation_path": "./playground/data/TEOChatlas/train/instruct_xBD.json"}
TEOCHATLAS_VIDEO_DATASET = [TEOCHATLAS_FMOW_DATASET, TEOCHATLAS_QFABRIC_DATASET, TEOCHATLAS_S2LOOKING_DATASET, TEOCHATLAS_XBD_DATASET]
for item in TEOCHATLAS_VIDEO_DATASET:
    item['data_path'] = "./playground/data/"
    item['dataset_type'] = 'teochat'

data_dict = {
    "vhm_dataset": VHM_DATASET,
    "refgeo_hbb": deepcopy([REFGEO_AVVG_DATASET, REFGEO_DIORRSVG_DATASET, REFGEO_GEOCHAT_DATASET, REFGEO_RSVG_DATASET, REFGEO_VRSBENCH_DATASET]),
    "refgeo_obb": deepcopy([REFGEO_AVVG_DATASET, REFGEO_GEOCHAT_DATASET, REFGEO_VRSBENCH_DATASET]),
    "refgeo_poly": deepcopy([REFGEO_AVVG_DATASET, REFGEO_GEOCHAT_DATASET, REFGEO_VRSBENCH_DATASET]),
    "coco_train2017": COCO_TRAIN2017_DATASET,
    "lrsvqa_train_visual_cot": LRSVQA_TRAIN_VISUAL_COT_DATASET,
    "lrsvqa_train_visual_cot_openended": LRSVQA_TRAIN_VISUAL_COT_OPENENDED_DATASET,
    "lrsvqa_train_visual_cot_mcq": LRSVQA_TRAIN_VISUAL_COT_MCQ_DATASET,
    "llava_onevision_vl": LLAVA_ONEVISION_VL_DATASET,
    "llava_onevision_text": LLAVA_ONEVISION_TEXT_DATASET,
    "processed_geochat_from_teochatlas": PROCESSED_GEOCHAT_FROM_TEOCHATLAS_DATASET,
    "teochatlas_fmow": TEOCHATLAS_FMOW_DATASET,
    "teochatlas_qfabric": TEOCHATLAS_QFABRIC_DATASET,
    "teochatlas_s2looking": TEOCHATLAS_S2LOOKING_DATASET,
    "teochatlas_xbd": TEOCHATLAS_XBD_DATASET,
    "teochatlas_video": TEOCHATLAS_VIDEO_DATASET,

    # dense det
    "dota_trainval1024": {**DOTA_TRAINVAL1024_DATASET, "dataset_type": "dense_det"},
    "dota_trainval512": {**DOTA_TRAINVAL512_DATASET, "dataset_type": "dense_det"},
    "dota_poly_trainval1024": {**DOTA_TRAINVAL1024_DATASET, "dataset_type": "dense_det_poly_only"},
    "dota_poly_trainval512": {**DOTA_TRAINVAL512_DATASET, "dataset_type": "dense_det_poly_only"},
    "dota_hbb_trainval1024": {**DOTA_TRAINVAL1024_DATASET, "dataset_type": "dense_det_hbb_only"},
    "dota_hbb_trainval512": {**DOTA_TRAINVAL512_DATASET, "dataset_type": "dense_det_hbb_only"},
    "fair1m_train682": {**FAIR1M_TRAIN682_DATASET, "dataset_type": "dense_det"},
    "fair1m_poly_train682": {**FAIR1M_TRAIN682_DATASET, "dataset_type": "dense_det_poly_only"},
    "dior_trainval800": {**DIOR_TRAINVAL800_DATASET,"dataset_type": "dense_det"},
    "dior_poly_trainval800": {**DIOR_TRAINVAL800_DATASET, "dataset_type": "dense_det_poly_only"},
    "srsdd_train1024": {**SRSDD_TRAIN1024_DATASET, "dataset_type": "dense_det"},
    "srsdd_poly_train1024": {**SRSDD_TRAIN1024_DATASET, "dataset_type": "dense_det_poly_only"},
    "rsar_trainval190to1000": {**RSAR_TRAINVAL190TO1000_DATASET, "dataset_type": "dense_det"},
    "rsar_poly_trainval190to1000": {**RSAR_TRAINVAL190TO1000_DATASET, "dataset_type": "dense_det_poly_only"},
    "bh_pools_train": {**BH_POOLS_DATASET, "dataset_type": "dense_det"},
    "bh_pools_poly_train": {**BH_POOLS_DATASET, "dataset_type": "dense_det_poly_only"},
    "bh_watertanks_train": {**BH_WATERTANKS_DATASET, "dataset_type": "dense_det"},
    "bh_watertanks_poly_train": {**BH_WATERTANKS_DATASET, "dataset_type": "dense_det_poly_only"},
    "HRSC2016_train": {**HRSC2016_DATASET, "dataset_type": "dense_det"},
    "HRSC2016_poly_train": {**HRSC2016_DATASET, "dataset_type": "dense_det_poly_only"},
}

for cfg in data_dict['refgeo_hbb']:
    cfg['dataset_type'] = 'refgeo_hbb_only'

for cfg in data_dict['refgeo_obb']:
    cfg['dataset_type'] = 'refgeo_obb_only'

for cfg in data_dict['refgeo_poly']:
    cfg['dataset_type'] = 'refgeo_poly_only'


def parse_sampling_rate(dataset_name):
    # check `%`
    match = re.search(r"%([\d\.]+)$", dataset_name)
    if match:
        ratio = float(match.group(1))
        assert 0 < ratio <= 100, f"Sampling rate must be in (0, 1], got {ratio}"
        return re.sub(r"%([\d\.]+)$", "", dataset_name), ratio / 100.0
    
    # check `*`
    match = re.search(r"\*(\d+)$", dataset_name)
    if match:
        return re.sub(r"\*(\d+)$", "", dataset_name), int(match.group(1))
    return dataset_name, 1


def get_data_config_list(dataset_names: Union[str, list[str]]):
    config_list = []

    if isinstance(dataset_names, str):
        if dataset_names.startswith('all'):
            sampling_msg = dataset_names.replace('all', '')
            dataset_names = [n + sampling_msg for n in data_dict.keys()]
        else:
            dataset_names = [dataset_names]
    
    for dataset_name in dataset_names:
        dataset_name, sampling_rate = parse_sampling_rate(dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy() # find corresponding data configs
            # synonyms_dict = category_synonyms.get(dataset_name, None)
            if not isinstance(config, list):
                config = [config]
            for cfg in config:
                cfg["sampling_rate"] = sampling_rate
                # if synonyms_dict:
                #     cfg['synonyms_dict'] = synonyms_dict
                cfg["dataset_name"] = dataset_name
                config_list.append(cfg)
        else:
            raise ValueError(f"do not find {dataset_name}")
            
    return config_list


def register_custom_sft_dataset(data_path, image_folder):
    if isinstance(data_path, str):
        data_path = [data_path]
    if isinstance(image_folder, str):
        image_folder = [image_folder]
    assert len(data_path) == len(image_folder), f"data_path: {data_path}, image_folder: {image_folder}"

    config = {}
    for i in range(len(data_path)):
        config[f"custom_sft_dataset_{i}"] = {
            "annotation_path": data_path[i],
            "data_path": image_folder[i],
        }
    data_dict.update(config)
    return list(config.keys())


def parse_config_file(config_file):
    if config_file.endswith(".json"):
        with open(config_file, "r") as f:
            config = json.load(f)
    elif config_file.endswith(".yaml"):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_file}")
    
    if isinstance(config, list):
        off_the_shelf_configs = [cfg for cfg in config if isinstance(cfg, str)]
        custom_configs = [cfg for cfg in config if not isinstance(cfg, str)]
        custom_configs = {f"custom_dataset_{i}": cfg for i, cfg in enumerate(custom_configs)}

    data_dict.update(custom_configs)
    return list(custom_configs.keys()) + off_the_shelf_configs


if __name__ == "__main__":
    dataset_names = ["vhm_dataset%50", "fair1m_train682", "fair1m_poly_train682", "dota_poly_trainval512"] # "vhm_dataset%50", "vhm_dataset"
    configs = get_data_config_list(dataset_names)
    for config in configs:
        print(config)
