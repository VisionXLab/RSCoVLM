import warnings
from typing import Optional, Union
from dataclasses import dataclass, field

from transformers import TrainingArguments, HfArgumentParser

# NOTE that the params.py is always called firstly, so the constants are pre-defined by qwen_vl_utils, not affected by the hacking of setting custom min/max_pixels.
from qwen_vl_utils.vision_process import MIN_PIXELS, MAX_PIXELS


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")


@dataclass
class TrainingArguments(TrainingArguments):
    # optim
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    # freeze
    freeze_vision: bool = field(default=False)
    freeze_language: bool = field(default=False)
    freeze_merger: bool = field(default=True)
    # lora
    lora_enable: bool = False
    use_dora: bool = False
    vision_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias_for_lora: str = "none"
    init_lora_weights: str = field(default='true')
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1


@dataclass
class DataArguments:
    # determine what dataset to use
    datasets: Optional[list[str]] = field(default=None, metadata={"nargs": "+"})
    data_path: Optional[list[str]] = field(default=None, metadata={"nargs": "+"})
    image_folder: Optional[list[str]] = field(default=None, metadata={"nargs": "+"})
    # visual length limit
    min_pixels: Optional[int] = field(default=MIN_PIXELS)  # 4 * 28 * 28
    max_pixels: Optional[int] = field(default=MAX_PIXELS)  # 16384 * 28 * 28
    # efficiency methods
    flatten_data: bool = field(default=False)  # maybe follow https://huggingface.co/blog/zh/packing-with-FA2 and https://arxiv.org/pdf/2407.09105
    packing_data: bool = field(default=False)
    packing_workers: Optional[int] = field(default=32)
    packing_interval: Optional[int] = field(default=64)
    dataset_randomness: Optional[int] = field(default=42)
    packing_cache: Optional[str] = field(default=None)
    # visual augmentation
    prob_random_resize: Optional[float] = field(default=0.0)
    # box response format
    prob_proxy_prompt: Optional[float] = field(default=0.5)
    prob_plain_text_prompt: Optional[float] = field(default=1.0)
    keep_empty_gt: Optional[bool] = field(default=True)
    # disable visual cot (for comparison)
    disable_visual_cot: Optional[bool] = field(default=False)
    # max sequence length and manner to process exceeded param
    max_length: Optional[int] = field(default=-100)  # also used as max packed length
    process_exceed: Optional[str] = field(default='truncate')
    # data sampling
    data_sampling_seed: Optional[int] = field(default=None)


def get_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.lora_enable and not training_args.freeze_language:
        raise ValueError("If `lora_enable` is True, `freeze_language` must also be True.")
    
    if training_args.vision_lora and not training_args.freeze_vision:
        raise ValueError("If `vision_lora` is True, `freeze_vision` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.init_lora_weights.lower() in ["true", "false"]:
        training_args.init_lora_weights = True if training_args.init_lora_weights.lower() == "true" else False

    if data_args.datasets is not None and len(data_args.datasets) == 1:
        data_args.datasets = data_args.datasets[0]
    
    if data_args.data_path is not None and len(data_args.data_path) == 1:
        data_args.data_path = data_args.data_path[0]

    if data_args.image_folder is not None and len(data_args.image_folder) == 1:
        data_args.image_folder = data_args.image_folder[0]

    if data_args.packing_data:
        if not data_args.flatten_data:
            warnings.warn("--flatten_data should be True when packing data is enabled, it has been automatically set.")
        data_args.flatten_data = True
        # accelerator_config = training_args.accelerator_config
        # assert not accelerator_config.dispatch_batches and not accelerator_config.split_batches, \
        #     "Set --accelerator_config \"{\\\"dispatch_batches\\\": false, \\\"split_batches\\\": false}\" to avoid batch splitting. "
        # training_args.do_not_prepare_dataloader_with_accelerator = True

    return model_args, data_args, training_args
