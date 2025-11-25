from functools import partial
from contextlib import contextmanager
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

modeling_qwen2_5_vl._flash_attention_forward

@contextmanager
def monkey_patch_flash_attention_to_pass_position_ids(position_ids):
    origin_flash_attention_forward = modeling_qwen2_5_vl._flash_attention_forward
    modeling_qwen2_5_vl._flash_attention_forward = partial(origin_flash_attention_forward, position_ids=position_ids)
    try:
        yield
    finally:
        modeling_qwen2_5_vl._flash_attention_forward = origin_flash_attention_forward