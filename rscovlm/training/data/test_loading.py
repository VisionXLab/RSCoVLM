import argparse
import random
from tqdm import tqdm
from collections import defaultdict

from transformers import Qwen2_5_VLProcessor

from qwen_vl_utils import process_vision_info
from rscovlm.training.data import make_supervised_data_module


def check_image(train_dataset, sample_num=None, disable_tqdm=False):
    if sample_num is not None:
        indices = random.sample(range(len(train_dataset)), sample_num)
    else:
        indices = range(len(train_dataset))

    for i in tqdm(indices, disable=disable_tqdm):
        messages = train_dataset.get_messages(i)
        process_vision_info(messages)


def test_check_loading_data(datasets_list=None, sample_num=None):
    if datasets_list is None:
        from rscovlm.training.data.config import data_dict
        datasets_list = list(data_dict.keys())

    processor = Qwen2_5_VLProcessor.from_pretrained("./playground/Qwen2.5-VL-3B-Instruct")

    errors = defaultdict(str)
    datasets = {}
    print("checking making datasets ...")
    for dataset in datasets_list:
        data_args = argparse.Namespace(datasets=[dataset])
        try:
            modules = make_supervised_data_module(processor, data_args)
            datasets[dataset] = modules["train_dataset"]
        except Exception as err:
            errors[dataset] += str(err) + "; "
            print(f"Error loading {dataset}: {err}")

    print(f"checking image ({'all' if sample_num is None else f'{sample_num} samples'}) ...")
    for dataset in datasets_list:
        try:
            check_image(datasets[dataset], sample_num=sample_num)
        except Exception as err:
            errors[dataset] += str(err) + "; "
            print(f"Error loading {dataset}: {err}")

    if len(errors) > 0:
        print("Errors occurred while loading the following datasets:")
        for dataset, err in errors.items():
            print(f"{dataset}: {err}")
        with open("tmp_training_data_error.log", "w") as f:
            for dataset, err in errors.items():
                f.write(f"{dataset}: {err}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None, nargs="+", help="datasets to check")
    parser.add_argument("--sample_num", type=int, default=None, help="sample number")
    args = parser.parse_args()

    test_check_loading_data(args.datasets, args.sample_num)
    