from transformers import Qwen2_5_VLProcessor
from rscovlm.training.data.processing_message import process_to_train_qwen2_5_vl_with_default_chat_template


model_path = "./playground/Qwen2.5-VL-3B-Instruct"
min_pixels=256 * 28 * 28 
max_pixels=1296 * 28 * 28
processor = Qwen2_5_VLProcessor.from_pretrained(
    model_path,
    min_pixels=min_pixels, 
    max_pixels=max_pixels,
    size={"shortest_edge": min_pixels, "longest_edge": max_pixels}
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "This is a test."}]},
]

text1 = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
output = process_to_train_qwen2_5_vl_with_default_chat_template(processor, messages)

import ipdb; ipdb.set_trace()
