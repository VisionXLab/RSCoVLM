import random
import warnings
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


IGNORE_INDEX = -100
PADDING_TO_MAX_LENGTH = False


def pad_position_ids(sequences, batch_first=False, padding_value=0, **kwargs):
    # [3, seq_len] --> [seq_len, 3]
    transposed = [seq.permute(1, 0) for seq in sequences]  # [seq_len, 3]
    padded = pad_sequence(transposed, batch_first=False, padding_value=padding_value, **kwargs)  # [seq_len, bs, 3]
    if batch_first:
        padded = padded.permute(1, 2, 0)  # [seq_len, bs, 3] --> [bs, 3, seq_len]
    else:
        padded = padded.permute(2, 1, 0)  # [seq_len, bs, 3] --> [3, bs, seq_len]
    return padded


class DataCollatorForQwen2_5_VL(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, max_length: Optional[int] = None, process_exceed: Optional[str] = 'truncate'):
        self.pad_token_id = pad_token_id
        self.max_length = max_length if max_length > 0 else None
        self.process_exceed = process_exceed

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_values_videos = []
        batch_image_thw = []
        batch_video_thw = []
        batch_second_per_grid_ts = []
        batch_position_ids = []
        batch_attention_mask = []
        
        for example in examples:
            keys = example.keys()
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            if "pixel_values_videos" in keys:
                batch_pixel_values_videos.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
            if "position_ids" in keys:
                batch_position_ids.append(example["position_ids"])
            if "attention_mask" in keys:
                batch_attention_mask.append(example["attention_mask"])

        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_side='right', padding_value=self.pad_token_id)
        labels = pad_sequence(batch_label_ids, batch_first=True, padding_side='right', padding_value=IGNORE_INDEX)
        if len(batch_attention_mask) > 0:
            attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_side='right', padding_value=False)  # TODO: maybe check padding_value=False
        else:
            attention_mask = input_ids != self.pad_token_id

        if self.max_length is not None:
            if PADDING_TO_MAX_LENGTH and len(input_ids[0]) < self.max_length:
                input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], self.max_length - len(input_ids[0])), self.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)], dim=1)
                labels = torch.cat([labels, torch.full((labels.shape[0], self.max_length - len(labels[0])), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], self.max_length - len(attention_mask[0])), False, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

            if self.process_exceed == 'truncate':
                input_ids = input_ids[:, :self.max_length]
                labels = labels[:, :self.max_length]
                attention_mask = attention_mask[:, :self.max_length]
            elif self.process_exceed == 'replace':
                seq_len = attention_mask.sum(dim=1)
                is_exceed = seq_len > self.max_length
                non_exceed_idx = torch.where(~is_exceed)[0]

                if len(non_exceed_idx) == 0:
                    warnings.warn("All sequences exceed the maximum length, hence, using `truncate` method in this batch.")
                    input_ids = input_ids[:, :self.max_length]
                    labels = labels[:, :self.max_length]
                    attention_mask = attention_mask[:, :self.max_length]
                elif len(non_exceed_idx) < len(seq_len):
                    idx_to_replace = non_exceed_idx.new_tensor(
                        random.choices(non_exceed_idx.tolist(), k=is_exceed.sum().item()))

                    input_ids[is_exceed] = input_ids[idx_to_replace]
                    input_ids = input_ids[:, :self.max_length]
                    labels[is_exceed] = labels[idx_to_replace]
                    labels = labels[:, :self.max_length]
                    attention_mask[is_exceed] = attention_mask[idx_to_replace]
                    attention_mask = attention_mask[:, :self.max_length]

            else:
                raise ValueError(f"Unsupported process_exceed method: {self.process_exceed}")

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_values_videos) > 0:
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_position_ids) > 0:
            position_ids = pad_position_ids(batch_position_ids, batch_first=False, padding_side='right', padding_value=1)

            if self.max_length is not None:
                if PADDING_TO_MAX_LENGTH and len(position_ids[0]) < self.max_length:
                    position_ids = torch.cat([position_ids, torch.full((position_ids.shape[0], position_ids.shape[1], self.max_length - len(position_ids[0])), 1, dtype=position_ids.dtype, device=position_ids.device)], dim=2)

                if self.process_exceed == 'truncate':
                    position_ids = position_ids[:, : , :self.max_length]
                elif self.process_exceed == 'replace':
                    if len(non_exceed_idx) == 0:
                        position_ids = position_ids[:, : , :self.max_length]
                    elif len(non_exceed_idx) < len(seq_len):
                        position_ids[is_exceed] = position_ids[idx_to_replace]
                else:
                    raise ValueError(f"Unsupported process_exceed method: {self.process_exceed}")

            data_dict["position_ids"] = position_ids

        return data_dict


def maybe_to_tensor(value, dtype=None):
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, (list, tuple)):
        data = torch.tensor(value, dtype=dtype)
        # print(f"len:{len(value)} data.shape:{data.shape} data.dtype:{data.dtype}")
        return data
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    else:
        raise TypeError(f"Unsupported type: {type(value)}")


class FlattenedDataCollatorForQwen2_5_VL(object):
    """Collate examples into packed sequence with multi-modal support."""

    def __init__(self, max_length: Optional[int] = None, process_exceed: Optional[str] = 'truncate'):
        self.max_length = max_length if max_length > 0 else None
        self.process_exceed = process_exceed

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_values_videos = []
        batch_image_thw = []
        batch_video_thw = []
        batch_second_per_grid_ts = []
        batch_position_ids = []
        batch_naive_position_ids = []
        
        for _examples in examples:
            if not isinstance(_examples, list):
                _examples = [_examples]
            for example in _examples:
                input_ids = maybe_to_tensor(example["input_ids"])
                batch_input_ids.append(input_ids)
                batch_label_ids.append(maybe_to_tensor(example["labels"]))
                batch_position_ids.append(maybe_to_tensor(example["position_ids"]))
                batch_naive_position_ids.append(torch.arange(len(input_ids), device=input_ids.device))
                if "pixel_values" in example:
                    batch_pixel_values.append(maybe_to_tensor(example["pixel_values"]))
                    batch_image_thw.append(maybe_to_tensor(example["image_grid_thw"]))
                if "pixel_values_videos" in example:
                    batch_pixel_values_videos.append(maybe_to_tensor(example["pixel_values_videos"]))
                    batch_video_thw.append(maybe_to_tensor(example["video_grid_thw"]))
                if "second_per_grid_ts" in example:
                    batch_second_per_grid_ts.extend(maybe_to_tensor(example["second_per_grid_ts"]))
                if "attention_mask" in example:
                    assert torch.all(maybe_to_tensor(example["attention_mask"])), example["attention_mask"]
        input_ids = torch.cat(batch_input_ids, dim=0)
        labels = torch.cat(batch_label_ids, dim=0)
        position_ids = torch.cat(batch_position_ids, dim=1)
        naive_position_ids = torch.cat(batch_naive_position_ids, dim=0)

        if self.max_length is not None:
            if self.process_exceed == 'truncate':
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                position_ids = position_ids[:, :self.max_length]
                naive_position_ids = naive_position_ids[:self.max_length]
            elif self.process_exceed == 'replace':
                seq_len = attention_mask.sum(dim=1)
                is_exceed = seq_len > self.max_length
                non_exceed_idx = torch.where(~is_exceed)[0]

                if len(non_exceed_idx) == 0:
                    warnings.warn("All sequences exceed the maximum length, hence, using `truncate` method in this batch.")
                    input_ids = input_ids[:, :self.max_length]
                    labels = labels[:, :self.max_length]
                    position_ids = position_ids[:, :self.max_length]
                    naive_position_ids = naive_position_ids[:self.max_length]
                elif len(non_exceed_idx) < len(seq_len):
                    idx_to_replace = non_exceed_idx.new_tensor(
                        random.choices(non_exceed_idx.tolist(), k=is_exceed.sum().item()))

                    input_ids[is_exceed] = input_ids[idx_to_replace]
                    input_ids = input_ids[:, :self.max_length]
                    labels[is_exceed] = labels[idx_to_replace]
                    labels = labels[:, :self.max_length]
                    position_ids[is_exceed] = position_ids[idx_to_replace]
                    position_ids = position_ids[:, :self.max_length]
                    naive_position_ids[is_exceed] = naive_position_ids[idx_to_replace]
                    naive_position_ids = naive_position_ids[:self.max_length]

            else:
                raise ValueError(f"Unsupported process_exceed method: {self.process_exceed}")

        data_dict = {
            'input_ids': input_ids.unsqueeze(0),
            'labels': labels.unsqueeze(0),
            'position_ids': position_ids.unsqueeze(1),
            'naive_position_ids': naive_position_ids.unsqueeze(0),
        }
        
        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_values_videos) > 0:
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
        return data_dict
    
