from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "./playground/Qwen2.5-VL-3B-Instruct"

llm = LLM(model=MODEL_PATH, enable_lora=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=16384,
    stop=["<im_end>"]
)

img_path = "./playground/data/coco/train2014/COCO_train2014_000000000049.jpg"
question = "Describe the image in detail"

messages = [[{
    "role": "user",
    "content": [
        {"type": "image", "image": f"file://{img_path}"},
        {"type": "text", "text": f"{question}"}
    ]
}]]

image_inputs, _ = process_vision_info(messages)
text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(text, image_inputs)]

outputs = llm.generate(
    inputs,
    sampling_params,
    lora_request=LoRARequest("adapter", 1, "output/qwen2-5-vl-ins_dota-hbb-trainval512_dora-eva-r64-a64_pix256-1296_lr0.0002-10epochs")
)
outputs_decoded = [o.outputs[0].text for o in outputs]
print(outputs_decoded)