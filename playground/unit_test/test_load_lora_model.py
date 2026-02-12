from peft import AutoPeftModel
from transformers import AutoProcessor

model_path = "./output/qwen2-5-vl-ins_pix64-1296"
model = AutoPeftModel.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model.peft_config['default'].base_model_name_or_path)

model = model.to("cuda")
model.eval()
inputs = processor.tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=5000)
print(processor.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

import ipdb; ipdb.set_trace()
