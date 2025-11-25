# import sys
# sys.path.insert(0, "/mnt/petrelfs/liqingyun/msr/code/rscovlm")
import os
import ast
import inspect
import logging
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import qwen_vl_utils

from rscovlm.utils import print_trainable_parameters
from .trainer import CustomTrainer
from .data import make_supervised_data_module
from .params import get_args

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_rscovlm/"

logger = logging.getLogger(__name__)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        logger.info(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def initialize_model_and_processor(model_args, training_args, data_args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=(torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
        attn_implementation="flash_attention_2", 
        use_cache=False,  # model.config.use_cache = False
    )

    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision)
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_language)  # this should be aligned with input embeddings
    set_requires_grad(model.model.parameters(), not training_args.freeze_language)

    if training_args.lora_enable:
        # prepare for lora
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=training_args.lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.bias_for_lora,
            init_lora_weights=training_args.init_lora_weights,
            use_dora=training_args.use_dora,
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    processor_kwargs = {"use_fast": True}
    # NOTE we should pay attention to the confusing logic of smart resizing of Qwen2.5-VL
    # There are two places where the input image is resized using min_pixels/max_pixels:
    # 1. qwen_vl_utils.process_vision_info: resize with the constant values MIN_PIXELS/MAX_PIXELS defined in qwen_vl_utils.vision_process.
    # 2. Qwen2VLImageProcessor: resize with the `size` dict.
    # But we may consider that pass a min_pixels/max_pixels to the AutoProcessor.from_pretrained is enough. But they actually not impact the both procedures.
    # Especially in the processor, the resize actually uses the `size` dict, which may be different from the min_pixels/max_pixels.
    # When you pass the min_pixels/max_pixels in AutoProcessor.from_pretrained, the ImageProcessingMixin.from_dict() firstly build the image processor with the default settings, then override the min_pixels/max_pixels with the passed values.
    # But if the `size` dict is not passed, it will not be overridden, so the resize in the processor will still use the default settings.
    # This will result in mismatching between training and inference, because you may train the model using the default settings, but in inference period, the min_pixels/max_pixels will be in the config file and impact the resizing.
    if data_args.min_pixels is not None:
        qwen_vl_utils.vision_process.MIN_PIXELS = data_args.min_pixels
        processor_kwargs["min_pixels"] = data_args.min_pixels
        processor_kwargs["size"] = {'shortest_edge': data_args.min_pixels}
    if data_args.max_pixels is not None:
        qwen_vl_utils.vision_process.MAX_PIXELS = data_args.max_pixels
        processor_kwargs["max_pixels"] = data_args.max_pixels
        processor_kwargs["size"] = processor_kwargs.get("size", {})
        processor_kwargs["size"].update({'longest_edge': data_args.max_pixels})

    processor = AutoProcessor.from_pretrained(model_args.model_id, **processor_kwargs)
    
    peak_mem = torch.cuda.max_memory_allocated()
    logger.info(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")
    return model, processor


def get_trainer(model_args, data_args, training_args):
    # prepare model and processor
    model, processor = initialize_model_and_processor(model_args, training_args, data_args)

    # prepare data
    data_module = make_supervised_data_module(processor=processor, data_args=data_args, model_for_position_ids=model)
    
    trainer = CustomTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )
    return trainer


def train():
    model_args, data_args, training_args = get_args()    
    trainer = get_trainer(model_args, data_args, training_args)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    train()
    