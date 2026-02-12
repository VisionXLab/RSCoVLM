# Data

We maintain the data collection in [this repo at HuggingFace Hub Platform](https://huggingface.co/datasets/Qingyun/remote-sensing-sft-data), which hosts all the training&evaluation data for RSCoVLM.

To make it easier for users to download and use, we have uploaded all the processed images and annotations. 
We recommend downloading the entire dataset and extracting it, as we strive to make it ready for use immediately. 
If you already have some of the images or annotations locally, you can exclude certain files during the download to save time. 
We have compressed the images and annotations for each dataset separately to facilitate such convenience.

If you encounter any issues, such as errors in the data or have other questions about the dataset, feel free to contact me via [GitHub issues](https://github.com/VisionXLab/RSCoVLM/issues)(prefered), or email me directly. 
We will continue to maintain the dataset.

## Downloading Guide

First, read the [Statement and ToU](https://github.com/VisionXLab/RSCoVLM/blob/master/Data.md#statement-and-tou), if aggred, go to [our dataset page](https://huggingface.co/datasets/Qingyun/remote-sensing-sft-data) and click the following button to access the dataset, your request will be approved automatically. NOTE once you request means that you agree with our ToU.

![request](TODO)

You can download with your web browser on [the file page](https://huggingface.co/datasets/Qingyun/remote-sensing-sft-data/tree/main).

We recommand downloading in terminal using huggingface_hub (`pip install --upgrade huggingface_hub`). You can refer to [the document](https://huggingface.co/docs/huggingface_hub/guides/download) for more usages.

```
# log in for huggingface account (if required, you can create your token at https://huggingface.co/settings/tokens)
hf login
# Set Huggingface Mirror for Chinese users (if required):
export HF_ENDPOINT=https://hf-mirror.com 
# Download the whole folder (you can also modify local-dir with your data path and make soft link here):
hf download Qingyun/remote-sensing-sft-data --repo-type dataset --local-dir ./playground/data
# If any error (such as network error) interrupts the downloading, you just need to execute the same command, the latest huggingface_hub will resume downloading.
```

If you already download some data, you can also exclude them to save time. For example, you can exclude DOTA(split_ss_dota) trainval images with the `--exclude` option. You can also only download certain file with the position arg `filenames` or the `--include` option.

```
# This will exclude the files and just download the others.
hf download Qingyun/remote-sensing-sft-data --repo-type dataset --local-dir ./playground/data --exclude **split_ss_dota_trainval**
# This will download the file and should put it in the folder.
hf download Qingyun/remote-sensing-sft-data split_ss_dota/trainval/split_ss_dota_trainval_annfiles.tar.gz --repo-type dataset --local-dir ./playground/data
# This will download the files and put them like the arrangement in the repo.
hf download Qingyun/remote-sensing-sft-data --repo-type dataset --local-dir ./playground/data --include **split_ss_dota_trainval**
```

Then, extract all files from the compressed files.

```
find . \( -name "*.tar.gz" -o -name "*.part0" \) -execdir bash -c '[[ "{}" =~ \.part0$ ]] && cat {} {}.part1 | tar -zxvf - || tar -zxvf {}' \;
```

At last, if required, you can delete all the compressed files.
```
# list the files to delete for checking (if required)
find . -type f -name "*.tar.gz*" -print
# delete
find . -type f -name "*.tar.gz*" -exec rm -f {} \;
```

## Data Format

We support both the LLaVA-style `conversations` format and Qwen-style openai-like `messages` format:

### `Conversations`
- Each training sample of the `conversations` format data is a data dict:

Your annotation file should follow one of the two formats below:

1. Single-image example (json or jsonl entry):

```json
{
  "image": "images/001.jpg",
  "source_dataset": "sub_dataset", 
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat's the main object in this picture?"
    },
    {
      "from": "gpt",
      "value": "A red apple on a wooden table"
    }
  ]
}

```

2. Multi-image example:
```json
{
  "images": ["cats/001.jpg", "cats/002.jpg"], 
  "source_dataset": ["sub_dataset1", "sub_dataset2"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n<image>\nWhat are the differences between these two cats?"
    },
    {
      "from": "gpt",
      "value": "The first cat is an orange tabby with short fur and green eyes, while the second is a gray Siamese with blue eyes and pointed coloration. They also appear to be in different environments - the first is indoors on a couch, the second is outdoors in a garden."
    }
  ]
}
```
> NOTE: Only local path is support for tha image.

### `messages`
- Each training sample of the `messages` format data is a data list:

1. Single-image example (json or jsonl entry):
```json
[
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

2. Multi-image example:
```json
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    },
    {"role": "assistant", "content": "They are the same."},
]
```

## Config data

We support three manners to let the program know what data you want to train with.

### pass `--image_folder` and `--data_path`

- You can pass the annotation json/jsonl to `--data_path` and the corresponding image folder path to `--image_folder`
- You can pass multiple `--image_folder` and `--data_path`, the number should be equal
- You can only use `SupervisedDatasetForQwen2_5_VL` in this mode.

### register as buildin datasets

You can register the usual datasets as buildin datasets

- Step 1: Register Your Dataset

Open the file: `rscoagent/training/data/config.py`

Add a dictionary describing your dataset, including both the annotation file and the image root path:

```python
YOUR_DATASET = {
    "annotation_path": "/absolute/path/to/your_dataset/annotations.json",
    "data_path": "/absolute/path/to/your_dataset/images/",
}
```
Then register it in data_dict:

```python
data_dict = {
    "your_dataset": YOUR_DATASET,
    # other pre-defined datasets...
}
```

- Step 2: pass the name of your dataset to `--datasets`
- You can pass multiple dataset names, a `ConcatDataset` may be built.
- Use "dataset_name%N" to sample N% of the data.
- For refgeo_* datasets, make sure the 'dataset_type' is set in data config.

### pass a json/yaml config file to `--datasets`

You can also write a json/yaml config file to be a replacement of the data config.

## Statement and ToU

We release the data under a CC-BY-4.0 license, with the primary intent of supporting research activities. 
We do not impose any additional using limitation, but the users must comply with the terms of use (ToUs) of the source dataset. 
This dataset is a processed version, intended solely for academic sharing by the owner, and does not involve any commercial use or other violations of the ToUs. 
Any usage of this dataset by users should be regarded as usage of the original dataset. 
If there are any concerns regarding potential copyright infringement in the release of this dataset, please contact me, and We will remove any data that may pose a risk.

## Cite

```
@ARTICLE{li2026rscovlm,
  author={Li, Qingyun and Ma, Shuran and Luo, Junwei and Yu, Yi and Zhou, Yue and Wang, Fengxiang and Lu, Xudong and Wang, Xiaoxing and He, Xin and Chen, Yushi and Yang, Xue},
  title={Co-Training Vision-Language Models for Remote Sensing Multi-Task Learning},
  journal={Remote Sensing},
  volume={18},
  year={2026},
  number={2},
  article-number={222},
  url={https://www.mdpi.com/2072-4292/18/2/222},
  issn={2072-4292},
  doi={10.3390/rs18020222}
}

@INPROCEEDINGS{11242725,
  author={Li, Qingyun and He, Xin and Shu, Xinya and Yu, Yi and Chen, Dong and Chen, Yushi and Yang, Xue},
  booktitle={IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={A Simple Aerial Detection Baseline of Multimodal Language Models}, 
  year={2025},
  pages={6833-6837},
  doi={10.1109/IGARSS55030.2025.11242725}
}
```

Please also cite the paper of the original source dataset if they are adopted in your research.
