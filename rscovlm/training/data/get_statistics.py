import argparse
import random
from tqdm import tqdm
from collections import defaultdict

from transformers import Qwen2_5_VLProcessor

from qwen_vl_utils import extract_vision_info
from rscovlm.training.data import make_supervised_data_module


def main(datasets_list=None):
    if datasets_list is None:
        from rscovlm.training.data.config import data_dict
        datasets_list = list(data_dict.keys())

    processor = Qwen2_5_VLProcessor.from_pretrained("./playground/Qwen2.5-VL-3B-Instruct")

    errors = defaultdict(str)
    datasets = {}
    print("making datasets ...")
    for dataset_name in datasets_list:
        data_args = argparse.Namespace(datasets=[dataset_name])
        try:
            modules = make_supervised_data_module(processor, data_args)
            datasets[dataset_name] = modules["train_dataset"]
        except Exception as err:
            errors[dataset_name] += str(err) + "; "
            print(f"Error loading {dataset_name}: {err}")

    sample_num = {
        dataset_name: len(datasets[dataset_name]) 
        for dataset_name in datasets_list
    }

    image_set = set()
    # text_token = defaultdict(int)

    # for dataset_name in datasets_list:
    #     dataset = datasets[dataset_name]

    #     for idx in tqdm(range(len(dataset)), desc=f"processing {dataset_name}"):
    #         messages = dataset.get_messages(idx)

    #         import ipdb; ipdb.set_trace()
    #         break

            # vision_info = extract_vision_info(messages)
            # for img_info in vision_info:
            #     image_or_video = img_info[img_info['type']]
                # if not isinstance(image_or_video, list):
                #     image_or_video = [image_or_video]
                # for _image_or_video in image_or_video:
                #     if not isinstance(_image_or_video, str):
                #         _image_or_video = str(_image_or_video)
                    # image_set[dataset_name].add(_image_or_video)

    
    
    image_set = image_set | set(meta['image'] for meta in datasets['processed_geochat_from_teochatlas'].list_data_meta)
    image_set = image_set | set(meta['image'] for meta in datasets['vhm_dataset'].list_data_meta)
    image_set = image_set | set(img for meta in datasets['teochatlas_video'].list_data_meta for img in meta['video'])
    image_set = image_set | set(f"llava_ov_{idx}" for idx in range(len(datasets['llava_onevision_vl%10'].list_data_meta)))
    image_set = image_set | set(meta['image_id'] for meta in datasets['refgeo_poly'].list_data_dict)
    image_set = image_set | set(meta['file_name'] for meta in datasets['dota_poly_trainval512'].list_data_dict)
    image_set = image_set | set(meta['image'] for meta in datasets['lrsvqa_train_visual_cot'].list_data_dict)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None, nargs="+", help="datasets to explore")
    args = parser.parse_args()

    main(args.datasets)
    