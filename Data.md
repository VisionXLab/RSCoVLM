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

## Statement and ToU

We release the data under a CC-BY-4.0 license, with the primary intent of supporting research activities. 
We do not impose any additional using limitation, but the users must comply with the terms of use (ToUs) of the source dataset. 
This dataset is a processed version, intended solely for academic sharing by the owner, and does not involve any commercial use or other violations of the ToUs. 
Any usage of this dataset by users should be regarded as usage of the original dataset. 
If there are any concerns regarding potential copyright infringement in the release of this dataset, please contact me, and We will remove any data that may pose a risk.

## Cite

```
@article{TODO}

@article{li2025lmmrotate,
  title={A Simple Aerial Detection Baseline of Multimodal Language Models},
  author={Li, Qingyun and Chen, YushWe and Shu, Xinya and Chen, Dong and He, Xin and Yu YWe and Yang, Xue },
  journal={arXiv preprint arXiv:2501.09720},
  year={2025}
}
```

Please also cite the paper of the original source dataset if they are adopted in your research.
