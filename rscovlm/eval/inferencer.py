import io
import os
import re
import json
import base64
import logging
import requests
from PIL import Image
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

import transformers
from accelerate.utils import send_to_device

import qwen_vl_utils
from qwen_vl_utils import smart_resize
from rscovlm.utils.qwen_vl_utils import process_vision_info, extract_vision_info


logger = logging.getLogger(__name__)


def check_weight(model):
    normal = True
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN weight found in {name}")
            normal = False
        if torch.isinf(param).any():
            print(f"Inf weight found in {name}")
            normal = False
    return normal


def get_torch_dtype(torch_dtype):
    if not isinstance(torch_dtype, str):
        return torch_dtype
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[torch_dtype]


def get_num_parameters(module):
    """Modified from print_trainable_parameters of peft"""
    def _get_parameter_numel(param):
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            # if using DS Zero 3 and the weights are initialized empty
            num_params = param.ds_numel
        return num_params
    
    if isinstance(module, torch.Tensor):  # nn.Parameter()
        num_params = _get_parameter_numel(module)
        return num_params if module.requires_grad else 0, num_params
        
    trainable_params = 0
    all_param = 0
    for param in module.parameters():
        num_params = _get_parameter_numel(param)
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


class BaseInferencer(metaclass=ABCMeta):

    model_type = None
    CONFIG_CLASS = transformers.AutoConfig
    MODEL_CLASS = transformers.AutoModel
    PROCESSOR_CLASS = transformers.AutoProcessor

    def __init__(self, 
                 model_ckpt_path, 
                 device, 
                 torch_dtype=None, 
                 attn_implementation="flash_attention_2",
                 wrap_with_ddp=False, 
                 max_max_length=16384, 
                 use_vllm=False,
                 lazy_loading=True):
        self.model_ckpt_path = model_ckpt_path
        self.device = device
        self.torch_dtype = get_torch_dtype(torch_dtype)
        self.attn_implementation = attn_implementation
        self.wrap_with_ddp = wrap_with_ddp
        self.max_max_length = max_max_length
        self.use_vllm = use_vllm
        self.model_loaded = False
        if not lazy_loading:
            self.initialize_model_and_processor()

    def initialize_model_and_processor(self):
        if self.torch_dtype is None:
            model_config = self.CONFIG_CLASS.from_pretrained(self.model_ckpt_path)
            self.torch_dtype = get_torch_dtype(model_config.torch_dtype)
        
        processor_kwargs = getattr(self, "processor_kwargs", {})
        model_kwargs = getattr(self, "model_kwargs", {})
        model_kwargs["attn_implementation"] = self.attn_implementation
        model_kwargs["torch_dtype"] = self.torch_dtype
        model_kwargs["device_map"] = {"": self.device}

        is_peft_model = os.path.exists(os.path.join(self.model_ckpt_path, "adapter_config.json"))
        if is_peft_model:
            if self.use_vllm:
                self.model, self.lora_request = self.initialize_peft_model_with_vllm(**model_kwargs)
            else:
                self.model = self.initialize_peft_model(**model_kwargs)
        else:
            if self.use_vllm:
                self.model = self.initialize_model_with_vllm(**model_kwargs)
            else:
                self.model = self.MODEL_CLASS.from_pretrained(self.model_ckpt_path, **model_kwargs).eval()

        if not self.use_vllm:
            if not check_weight(self.model):
                raise ValueError("NaN or Inf weight found in model")

        self.processor = self.PROCESSOR_CLASS.from_pretrained(self.model_ckpt_path, **processor_kwargs)

        if self.wrap_with_ddp:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])

        self.model_loaded = True

        if not self.use_vllm:
            self.print_model_info()

            # set max_length automatically, and set img_context_token_id for internvl2
            if self.model_type == "florence2":
                self.max_length = self.model_config.text_config.max_position_embeddings
            elif self.model_type in ["internvl2", "internvl_chat"]:
                self.model.img_context_token_id = self.processor.img_context_token_id
                self.max_length = self.model_config.llm_config.max_position_embeddings
            elif self.model_type == "qwen2_vl":
                self.max_length = self.model_config.max_position_embeddings
            elif self.model_type == "llava_qwen":
                self.max_length = getattr(
                    self.model_config, 
                    "max_position_embeddings", 
                    self.model_config.max_position_embeddings)
            elif self.model_type == "qwen2_5_vl":
                self.max_length = self.model_config.max_position_embeddings
            else:
                self.max_length = 32768
            self.max_length = min(self.max_length, self.max_max_length)
        else:
            self.max_length = self.max_max_length

    def initialize_model_with_vllm(self, **model_kwargs):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        model_kwargs = self.resolve_model_kwargs_for_vllm(model_kwargs)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=self.max_max_length, stop=["<im_end>"])
        if os.environ.get("VLLM_IMAGE_LIMIT", None) is not None:
            image_limit = int(os.environ.get("VLLM_IMAGE_LIMIT"))
            model_kwargs["limit_mm_per_prompt"] = {"image": image_limit}
        return LLM(model=self.model_ckpt_path, **model_kwargs)  # consider `gpu_memory_utilization=0.9, enable_prefix_caching=True`

    def initialize_peft_model(self, **model_kwargs):
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(self.model_ckpt_path)
        base_model = self.MODEL_CLASS.from_pretrained(config.base_model_name_or_path, **model_kwargs)
        return PeftModel.from_pretrained(base_model, self.model_ckpt_path).eval()

    def initialize_peft_model_with_vllm(self, **model_kwargs):
        from peft import PeftConfig
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        model_kwargs = self.resolve_model_kwargs_for_vllm(model_kwargs)
        config = PeftConfig.from_pretrained(self.model_ckpt_path)
        if os.environ.get("VLLM_IMAGE_LIMIT", None) is not None:
            image_limit = int(os.environ.get("VLLM_IMAGE_LIMIT"))
            model_kwargs["limit_mm_per_prompt"] = {"image": image_limit}
        base_model = LLM(model=config.base_model_name_or_path, enable_lora=True, **model_kwargs)  # consider `gpu_memory_utilization=0.9, enable_prefix_caching=True`
        lora_request = LoRARequest("adapter", 1, self.model_ckpt_path)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=self.max_max_length, stop=["<im_end>"])
        return base_model, lora_request

    @staticmethod
    def resolve_model_kwargs_for_vllm(model_kwargs):
        for kwarg_names in ('attn_implementation',):
            if kwarg_names in model_kwargs:
                del model_kwargs[kwarg_names]
        if 'torch_dtype' in model_kwargs:
            torch_dtype = model_kwargs.pop('torch_dtype')
            if torch_dtype is not None:
                model_kwargs['dtype'] = torch_dtype
        if 'device_map' in model_kwargs:
            device = model_kwargs.pop('device_map')[""]
            if device is not None:
                model_kwargs['device'] = device
        return model_kwargs

    def data_to_dtype(self, data_dict):
        for key, value in data_dict.items():
            if (isinstance(value, torch.Tensor) and torch.is_floating_point(value)) \
                or key in ("images", "image", "pixel_values"):
                data_dict[key] = value.to(self.torch_dtype)
        return data_dict

    @property
    def unwrapped_model(self):
        return self.model.module if self.wrap_with_ddp else self.model
    
    @property
    def model_config(self):
        return self.unwrapped_model.config

    def print_model_info(self):
        pass

    @torch.no_grad()
    @abstractmethod
    def __call__(self, data_pipe_list: list[dict]):
        """
        Inference model on a batch of data.

        data_pipe is a dictionary contains all the information of a sample.
        This function requires the `message` item and adds the `output`, `original_width`, `original_height`, 
        `resized_width`, and `resized_height` items to the dictionary.
        """
        if not self.model_loaded:
            self.initialize_model_and_processor()


class Qwen25VLInferencer(BaseInferencer):

    model_type = "qwen2_5_vl"

    def __init__(self, *args, min_pixels=None, max_pixels=None, mimo_no_think=False, **kwargs):
        self.CONFIG_CLASS = transformers.Qwen2_5_VLConfig
        self.MODEL_CLASS = transformers.Qwen2_5_VLForConditionalGeneration
        self.PROCESSOR_CLASS = transformers.Qwen2_5_VLProcessor
        self.processor_kwargs = {}
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.mimo_no_think = mimo_no_think
        super().__init__(*args, **kwargs)

    @property
    def min_pixels(self):
        return self._min_pixels
    
    @min_pixels.setter
    def min_pixels(self, value):
        self._min_pixels = value
        if value is not None:
            qwen_vl_utils.vision_process.MIN_PIXELS = value
            self.processor_kwargs["min_pixels"] = value
            self.processor_kwargs["size"] = self.processor_kwargs.get("size", {})
            self.processor_kwargs["size"].update({'shortest_edge': value})  # actually we do not really need this

    @property
    def max_pixels(self):
        return self._max_pixels
    
    @max_pixels.setter
    def max_pixels(self, value):
        self._max_pixels = value
        if value is not None:
            qwen_vl_utils.vision_process.MAX_PIXELS = value
            self.processor_kwargs["max_pixels"] = value
            self.processor_kwargs["size"] = self.processor_kwargs.get("size", {})
            self.processor_kwargs["size"].update({'longest_edge': value})  # actually we do not really need this

    def initialize_model_and_processor(self):
        super().initialize_model_and_processor()
        if self.min_pixels is None:
            self.min_pixels = self.processor.image_processor.min_pixels
        if self.max_pixels is None:
            self.max_pixels = self.processor.image_processor.max_pixels
    
    def print_model_info(self):
        _, all_param = get_num_parameters(self.unwrapped_model)
        _, vis_and_merger_param = get_num_parameters(self.unwrapped_model.visual)
        _, lang_param = get_num_parameters(self.unwrapped_model.model)
        logger.info(f"Model loaded from {self.model_ckpt_path}, "
                    f"with {all_param:,d} parameters, including {vis_and_merger_param:,d} "
                    f"for vision tower and multimodal merger, "
                    f"{lang_param:,d} for language model. "
                    f"Model max_pixels: {self.max_pixels}, min_pixels: {self.min_pixels}")

    @staticmethod
    def add_mimo_no_think(data_pipe_list):
        for sample in data_pipe_list:
            message = sample["message"]
            for msg in reversed(message):
                if msg["role"] != "user":
                    continue
                content = msg["content"]
                if isinstance(content, list):
                    for c in reversed(content):
                        if c["type"] == "text":
                            c["text"] += " /no_think"
                            break
                elif isinstance(content, str):
                    msg["content"] = content + " /no_think"
        return data_pipe_list

    @staticmethod
    def extract_thinking_content(text):
        pattern = r'^\s*<think>(.*?)</think>'
        match = re.match(pattern, text, re.DOTALL)
        
        if match:
            thinking_content = match.group(1).strip()
            cleaned_text = text[match.end():].strip()
            return thinking_content, cleaned_text
        else:
            return None, text

    @staticmethod
    def prepare_visual_cot_message(message, previous_output, image_inputs, image_grid_thw, factor):
        text = (
            previous_output
            .lstrip('</tool_call>\n')
            .rstrip('\n</tool_call>')
        )
        try:
            tool_call = json.loads(text)

            assert tool_call['name'] == 'image_zoom_in', tool_call
            bbox = tool_call['arguments']['bbox']

            image_grid_h, image_grid_w = image_grid_thw[0][1].cpu().item(), image_grid_thw[0][2].cpu().item()
            resized_h, resized_w = image_grid_h * factor, image_grid_w * factor
            w, h = image_inputs[0].size

            x1, y1, x2, y2 = bbox
            x1 = x1 * w / resized_w
            y1 = y1 * h / resized_h
            x2 = x2 * w / resized_w
            y2 = y2 * h / resized_h
            bbox = [x1, y1, x2, y2]

            message.extend([
                {"role": "assistant", "content": previous_output},
                {"role": "user", "content": [
                    {"type": "image", "image": image_inputs[0].crop(bbox)},
                    {"type": "text", "text": "Here's the zoomed-in image. You can complete your task now."}
            ]}])
        except Exception as e:
            print(e)
            message.extend([
                {"role": "assistant", "content": previous_output},
                {"role": "user", "content": [
                    {"type": "image", "image": image_inputs[0]},
                    {"type": "text", "text": "Here's the zoomed-in image. You can complete your task now."}
            ]}])
        return message

    def maybe_visual_cot(self, data_pipe_list, image_inputs, image_grid_thw, factor):
        # judge whether there are visual cot samples, if so, conduct visual cot
        image_start_idx = 0
        visual_cot_data_pipe_list = []
        for idx, sample in enumerate(data_pipe_list):
            output_txt = sample['output']
            if '<tool_call>' in output_txt:
                n_image = len(extract_vision_info(sample['message']))
                sample['message'] = self.prepare_visual_cot_message(
                    sample['message'], 
                    output_txt, 
                    image_inputs[image_start_idx:image_start_idx + n_image], 
                    image_grid_thw[image_start_idx:image_start_idx + n_image], 
                    factor
                )
                visual_cot_data_pipe_list.append(sample)
                image_start_idx += n_image
        
        if len(visual_cot_data_pipe_list) != 0:
            self(visual_cot_data_pipe_list)

        return data_pipe_list
    
    @torch.no_grad()
    def __call__(self, data_pipe_list):
        super().__call__(data_pipe_list)

        if self.mimo_no_think:
            self.add_mimo_no_think(data_pipe_list)

        batch_messages = [sample["message"] for sample in data_pipe_list]
        text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs, image_sizes = process_vision_info(batch_messages, return_image_sizes=True)  # TODO: refactor here

        if self.use_vllm:
            outputs = self.model.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(text, image_inputs)], 
                self.sampling_params, lora_request=getattr(self, 'lora_request', None)
            )
            batch_output_text = [o.outputs[0].text for o in outputs]
        else:
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = self.data_to_dtype(send_to_device({**inputs}, self.device))
            generated_ids = self.model.generate(**inputs, use_cache=True, max_new_tokens=self.max_length, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            batch_output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        for j, sample in enumerate(data_pipe_list):
            output = batch_output_text[j]
            thinking_content, output = self.extract_thinking_content(output)
            if thinking_content:
                sample["think"] = thinking_content
            sample["output"] = output
            if video_inputs is None and len(image_inputs) > 0:
                sample["original_width"], sample["original_height"], sample["resized_width"], sample["resized_height"] = image_sizes[j]  # TODO: 如果icl这种有多个图像的怎么办？这里似乎有问题

        if not self.use_vllm and 'image_grid_thw' in inputs:  # TODO: support vllm for visual cot
            data_pipe_list = self.maybe_visual_cot(data_pipe_list, image_inputs, inputs['image_grid_thw'], self.model_config.vision_config.spatial_patch_size)
        return data_pipe_list
    

class Qwen2VLInferencer(Qwen25VLInferencer):
    
    model_type = "qwen2_vl"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CONFIG_CLASS = transformers.Qwen2VLConfig
        self.MODEL_CLASS = transformers.Qwen2VLForConditionalGeneration
        self.PROCESSOR_CLASS = transformers.Qwen2VLProcessor
    

class BaseAPIInferencer(metaclass=ABCMeta):  # TODO

    def __init__(self, *args, api_name='gpt-4o', backends='request', max_max_length=None, **kwargs):
        self.api_name = api_name
        self.backends = backends
        self.max_tokens = max_max_length
        self.info = getattr(self, api_name)
        self.model_name = self.info.pop("model_name", None)

    def get_with_request(self, api_key, base_url, payload):
        if not hasattr(self, 'headers'):
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

        if not "model" in payload:
            response = requests.get(base_url.rstrip('/') + '/models', headers=self.headers)
            if response.status_code == 200:
                models = response.json()
                if models.get('data'):
                    model_name = models['data'][0]['id']
                else:
                    raise ValueError("No models found.")
            else:
                raise Exception(f"Error fetching models: {response.status_code} - {response.text}")
            payload["model"] = self.model_name = model_name

        response = requests.post(
            base_url.rstrip('/') + "/chat/completions",
            headers=self.headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        else:
            return response
        
    def get_with_openai(self, api_key, base_url, payload):
        if not hasattr(self, 'client'):
            from openai import OpenAI
            if api_key in ["", None]:
                api_key = "none"

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        if not "model" in payload:
            model_name = self.client.models.list().data[0].id
            payload["model"] = self.model_name = model_name

        chat_response = self.client.chat.completions.create(**payload)
        return chat_response
    
    @staticmethod
    def encode_image_base64(image_source):
        if not isinstance(image_source, str):  # pil
            with io.BytesIO() as output:
                image_source.save(output, format="JPEG")
                base64_image = base64.b64encode(output.getvalue()).decode("utf-8")
            return base64_image
        elif image_source.startswith("http://") or image_source.startswith("https://"):
            response = requests.get(image_source)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode("utf-8")
            else:
                raise ValueError(f"Image Download Failure: HTTP state: {response.status_code}")
        elif os.path.exists(image_source):
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            raise ValueError(f"invalid image path/url: {image_source}")
        
    def get_single_round_prompt_messages(
            self, image, prompt, sys_prompt=None, 
            min_pixels=512*28*28, max_pixels=2048*28*28,
        ):
        
        messages = []
        
        if sys_prompt is not None:
            messages.append({
                "role": "system",
                "content": [{"type":"text","text": sys_prompt}]}
            )
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "image_url": {"url": f"data:image;base64,{self.encode_image_base64(image)}"},
                },
                {"type": "text", "text": prompt},
            ],
        })
        return messages
    
    def get_payload(self, *args, messages=None, **kwargs):
        if messages is None:
            messages = self.get_single_round_prompt_messages(*args, **kwargs)
        payload = {"messages": messages}
        if self.model_name is not None:
            payload["model"] = self.model_name
        if getattr(self, 'max_tokens', None) is not None:
            payload["max_tokens"] = self.max_tokens
        if getattr(self, 'top_p', None) is not None:
            payload["top_p"] = self.top_p
        payload["temperature"] = getattr(self, 'temperature', 0)
        return payload
    
    def get_res(self, payload):
        api_key = self.info["api_key"]
        base_url = self.info["base_url"]
        func = getattr(self, f'get_with_{self.backends}')
        return func(api_key, base_url, payload)
    
    def encode_message_images(self, messages):
        def _encode(item):
            if isinstance(item, str) and item.startswith("data:image/") and 'base64' in item:
                return item
            elif isinstance(item, str) and (item.startswith("http://") or item.startswith("https://")):
                return f"data:image;base64,{self.encode_image_base64(item)}"
            elif isinstance(item, str) and item.startswith("file://"):
                return f"data:image;base64,{self.encode_image_base64(item[7:])}"
            elif isinstance(item, Image.Image):
                return f"data:image;base64,{self.encode_image_base64(item)}"
            elif isinstance(item, list):
                return [_encode(i) for i in item]
            elif isinstance(item, dict):
                if item.get('type') == 'image' and 'image' in item:
                    item['type'] = 'image_url'  # TODO: qwen may also allow 'image_url' type
                    item['image_url'] = {"url": item.pop('image')}
                    if hasattr(self, 'image_kwargs'):
                        item.update(self.image_kwargs)
                return {k: _encode(v) for k, v in item.items()}
            else:
                return item
        return _encode(messages)

    @abstractmethod
    def __call__(self, data_pipe_list: list[dict]):
        """
        Inference model on a batch of data.

        data_pipe is a dictionary contains all the information of a sample.
        This function requires the `message` item and adds the `output`, `original_width`, `original_height`, 
        `resized_width`, and `resized_height` items to the dictionary.
        """
        if not self.model_loaded:
            self.initialize_model_and_processor()
    

class Qwen25VLAPIInferencer(BaseAPIInferencer):
    qwen_2_5_vl_3b = {
        'api_key': 'none',
        'base_url': 'http://10.140.60.20:8082/v1',
    }  # vllm QwenVL2.5 3B
    qwen_2_5_vl_7b = {
        'api_key': 'none',
        'base_url': "http://10.140.60.20:8083/v1",
    }  # vllm QwenVL2.5 7B
    qwen_2_5_vl_72b = {
        'api_key': 'none',
        'base_url': 'http://10.140.52.49:10000/v1',
    }  # vllm -> nginx QwenVL2.5 72B
    qwen_2_5_vl_max = {
        'api_key': 'sk-9010fae9707444faa7d307480e572460',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model_name': 'qwen-vl-max-2025-01-25',
    }  # claudeshop api (not support latest qwen-vl-max)  # TODO: maybe the claudeshop api has been updated
    # min/max_pixels # NOTE that vllm may not support min_pixels and max_pixels
    def __init__(self, *args, min_pixels=None, max_pixels=None, **kwargs):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_kwargs = {}
        if min_pixels is not None:
            self.image_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            self.image_kwargs["max_pixels"] = max_pixels
        super().__init__(*args, **kwargs)

    @property
    def min_pixels(self):
        return self._min_pixels
    
    @min_pixels.setter
    def min_pixels(self, value):
        self._min_pixels = value
        if value is not None:
            qwen_vl_utils.vision_process.MIN_PIXELS = value

    @property
    def max_pixels(self):
        return self._max_pixels
    
    @max_pixels.setter
    def max_pixels(self, value):
        self._max_pixels = value
        if value is not None:
            qwen_vl_utils.vision_process.MAX_PIXELS = value

    def get_image_sizes(self, data_pipe_list):
        batch_messages = [sample["message"] for sample in data_pipe_list]
        _, _, image_sizes = process_vision_info(batch_messages, return_image_sizes=True)
        return image_sizes

    def __call__(self, data_pipe_list):
        image_sizes = self.get_image_sizes(data_pipe_list)
        for data_pipe, image_size in zip(data_pipe_list, image_sizes):
            # we only support batch_size=1 actually
            message = self.encode_message_images(data_pipe["message"])
            payload = self.get_payload(messages=message)
            res = self.get_res(payload)
            output_txt = res['choices'][0]['message']['content']
            data_pipe["output"] = output_txt
            data_pipe["original_width"], data_pipe["original_height"], data_pipe["resized_width"], data_pipe["resized_height"] = image_size  # TODO: 如果icl这种有多个图像的怎么办？这里似乎有问题
        return data_pipe_list


def get_inferencer(model_name_of_path, *args, **kwargs):
    if model_name_of_path.startswith("api:"):  # e.g., api:qwen_2_5_vl_72b
        return Qwen25VLAPIInferencer(api_name=model_name_of_path[4:], *args, **kwargs)

    if os.path.exists(model_name_of_path):
        is_peft_model = os.path.exists(os.path.join(model_name_of_path, "adapter_config.json"))
        if is_peft_model:
            p = json.load(open(os.path.join(model_name_of_path, "adapter_config.json"), "r"))["base_model_name_or_path"]
        else:
            p = model_name_of_path
        model_type = json.load(open(os.path.join(p, "config.json"), "r"))["model_type"]
    else:  # maybe not support peft model? TODO: try and support peft model on hub
        model_type = transformers.AutoConfig.from_pretrained(model_name_of_path).model_type

    if model_type == "qwen2_5_vl":
        return Qwen25VLInferencer(model_name_of_path, *args, **kwargs)
    elif model_type == "qwen2_vl":
        return Qwen2VLInferencer(model_name_of_path, *args, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # model_path = "./playground/Qwen2.5-VL-3B-Instruct"
    model_path = "output/qwen2-5-vl-ins_dota-poly-trainval512_lora-r64-a64_pix256-1296_lr0.0002-10epochs/merged"
    model = Qwen25VLInferencer(model_path, 'cuda', use_vllm=True, torch_dtype='bf16')
    model.initialize_model_and_processor()

    img_path = "./playground/data/coco/train2014/COCO_train2014_000000000049.jpg"
    question = "Describe the image in detail"

    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": f"{question}"}
        ]
    }]
    data_pipe = {'message': message}
    print(model([data_pipe]))

    import ipdb; ipdb.set_trace()
