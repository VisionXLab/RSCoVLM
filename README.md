<p align="center">
  <h1 align="center">RSCoVLM: Co-Training Vision Language Models for Remote Sensing Multi-task Learning</h1>
  <p align="center">
      <a href='https://scholar.google.com/citations?user=TvsTun4AAAAJ' style='text-decoration: none' >Qingyun Li*</a><sup></sup>&emsp;
      <a href='https://ieeexplore.ieee.org/author/37089353293' style='text-decoration: none' >Shuran Ma*</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=6XibZaYAAAAJ&hl' style='text-decoration: none' >Junwei Luo*</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=OYtSc4AAAAAJ' style='text-decoration: none' >Yi Yu*</a><sup></sup>&emsp;
      <a href='https://zytx121.github.io/' style='text-decoration: none' >Yue Zhou</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=1xkMiSMAAAAJ' style='text-decoration: none' >Fengxiang Wang</a><sup></sup>&emsp;
      <a href='https://lucky-lance.github.io/' style='text-decoration: none' >Xudong Lu</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=n2ewxUIAAAAJ' style='text-decoration: none' >Xiaoxing Wang</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=WQgE8l8AAAAJ' style='text-decoration: none' >Xin He</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=A39S7JgAAAAJ' style='text-decoration: none' >Yushi Chen</a><sup></sup>&emsp;
      <a href='https://yangxue0827.github.io/' style='text-decoration: none' >Xue Yang</a><sup></sup>&emsp;
      <a href='https://thinklab.sjtu.edu.cn/' style='text-decoration: none' >Junchi Yan</a><sup></sup>&emsp;
      <div align="center">
      <a href='https://github.com/VisionXLab'><img src='https://img.shields.io/badge/arXiv-2501.09720-brown.svg?logo=arxiv&logoColor=white'></a>
      <a href='https://github.com/VisionXLab'><img src='https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=HuggingFace&logoColor=white'></a>
	  </div>
    <p align='center'>
        If you find our work helpful, please consider giving us a ⭐!
    </p>
   </p>
</p>

This repo is a technical practice of using **R**emote **S**ensing data to **Co**llaboratively train Large **V**ision **L**anguage **M**odels and hosts the official implementation of the paper: **Co-Training Vision Language Models for Remote Sensing Multi-task Learning**.

## Get Started

First, refer to [Enviroment.md](Enviroment.md) to prepare an enviroment.

For training rscovlm, firstly refer to [Data.md](Data.md) to prepare/download the data.

> NOTE:
> We support multi-nodes distributed training based on torchrun. If your resource platform is different and requires multi-nodes distributed training, you may need adapt the shell scripts to your platform. Or you can mult the node count to gradient_accumulation_steps option. Concat us in [issue](https://github.com/VisionXLab/RSCoVLM/issues) for more support.

TODO: add instructions of practices and interfaces

### Update to the latest version

We may update the codebase, the commit log is [here](https://github.com/VisionXLab/RSCoVLM/commits/master/).

If you have installed the previous version and would like to update to the latest version, you can:
```
cd RSCoVLM
git pull origin master  # if you have modification, commit your changes and merge the branches
pip install -e .
```

## Contact and Acknowledge

Feel free to contact me through my email (21b905003@stu.hit.edu.cn) or [github issue](https://github.com/VisionXLab/RSCoVLM/issues). I'll continue to maintain this repo.

The code is based on [Transformers](https://github.com/huggingface/transformers) and [MMRotate](https://github.com/open-mmlab/mmrotate). Many modules refer to [InternVL](https://github.com/OpenGVLab/InternVL) and [LLaVA](https://github.com/haotian-liu/LLaVA). The model architecture benefits from the open-source general-purpose vision-language model [Qwen-VL series](https://github.com/QwenLM/Qwen3-VL). Thanks for their brilliant works.

## Citation

If you find our paper or benchmark helpful for your research, please consider citing our paper and giving this repo a star ⭐. Thank you very much!

> The manuscript has been submmitted at 2026.11.14, and has not been accepted.

```bibtex
@article{TODO}

@article{li2025lmmrotate,
  title={A Simple Aerial Detection Baseline of Multimodal Language Models},
  author={Li, Qingyun and Chen, Yushi and Shu, Xinya and Chen, Dong and He, Xin and Yu Yi and Yang, Xue },
  journal={arXiv preprint arXiv:2501.09720},
  year={2025}
}
```
