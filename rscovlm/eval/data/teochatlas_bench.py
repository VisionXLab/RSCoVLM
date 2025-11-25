import os
import json

from rscovlm.training.data.sft_dataset import get_messages_from_llava_style_conversations
from rscovlm.training.data.xeochat_dataset import process_teochatlas_messages, compare_box_in_two_texts, remove_box_from_text


def split_messages_and_gt(message_with_gt):
    message = []
    gt = []
    for msg in message_with_gt:
        if msg['role'] == 'user':
            message.append(msg)
        elif msg['role'] == 'assistant':
            gt.append(msg)
    assert len(gt) == 1, gt

    gt_content = gt[0]['content']
    if isinstance(gt_content, list):
        assert len(gt_content) == 1, gt_content
        gt_content = gt_content[0]['text']
    return message, gt_content


def process_teochatlas_video_path(video_item):
    for i in range(len(video_item)):
        video_item[i] = video_item[i].replace('TEOChatlas/eval/', '')
    return video_item


def maybe_debug(x):
    conversations = x['conversations']
    has_bug = False

    for conv in conversations:
        if "from" not in conv:
            has_bug = True
            break

    if has_bug:
        assert len(conversations) == 2, conversations

        if 'from' not in conversations[0]:
            conversations[0]['from'] = "human"
        else:
            assert conversations[0]['from'] == "human", conversations[0]

        if 'from' not in conversations[1]:
            conversations[1]['from'] = "gpt"
        else:
            assert conversations[1]['from'] == "gpt", conversations[1]

    x['conversations'] = conversations
    return x


def maybe_debug_again(x):
    # for region_captioning task, there is repeated bbox in the response, we need to remove them
    if compare_box_in_two_texts(x['conversations'][0]['value'], x['conversations'][1]['value']):
        x['conversations'][1]['ori_value'] = x['conversations'][1]['value']
        x['conversations'][1]['value'] = remove_box_from_text(x['conversations'][1]['value'])
    return x


def _prepare_teochatlas_eval(data_root, ann_filename, prompt_type=None, min_pixels=None, max_pixels=None):
    data_root = os.path.join(data_root, 'eval')
    ann_filepath = os.path.join(data_root, ann_filename)

    with open(ann_filepath, 'r') as f:
        data = json.load(f)

    for idx, x in enumerate(data):
        assert len(x['conversations']) == 2, x

        x['data_path'] = data_root
        x['video'] = process_teochatlas_video_path(x['video'])

        x = maybe_debug(x)
        x = maybe_debug_again(x)
        message_with_gt = get_messages_from_llava_style_conversations(x)
        message_with_gt = process_teochatlas_messages(message_with_gt, min_pixels, max_pixels, use_json_prompt=(prompt_type=="json"))
        message, ground_truth = split_messages_and_gt(message_with_gt)

        x['ground_truth'] = ground_truth
        x['message'] = message
        x['idx'] = idx
    return data


def _prepare_teochatlas_geochat_eval(data_root, ann_filename):
    data_root = os.path.join(data_root, 'eval')
    ann_filepath = os.path.join(data_root, ann_filename)

    with open(ann_filepath, 'r') as f:
        data = json.load(f)

    for idx, x in enumerate(data):
        image = process_teochatlas_video_path(x['video'])[0]
        image_path = os.path.join(data_root, image)

        conversations = x['conversations']
        assert len(conversations) == 2, conversations

        question = conversations[0]['value'].replace('This is a satellite image: <video> ', '<image>').strip()
        ground_truth = conversations[1]['value']

        message = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]},
        ]

        x['ground_truth'] = ground_truth
        x['message'] = message
        x['idx'] = idx
    return data


teochatlas_temporal_files = (
    'S2Looking_SRE_QA.json', 
    'fMoW_Low_Res.json', 
    'S2Looking_RQA.json', 
    'fMoW_High_Res.json', 
    'CDVQA.json', 
    'xBD_SRE_QA_RQA.json', 
    'xBD_Change_Detection_Localization.json', 
    'QFabric_TRE_RTQA.json', 
    'QFabric_RQA2.json', 
    'xBD_Change_Detection_Classification.json', 
    'QFabric_RQA5_RTQA5.json', 
    'ABCD.json', 
    'S2Looking_Change_Detection.json',
)


def prepare_teochatlas_eval(data_root, benchmark, prompt_type=None, min_pixels=None, max_pixels=None):
    if benchmark == 'teochatlas_geochat':
        data_pipe_list = []
        for filename in (
            'AID.json',
            'UCMerced.json',
            'HRBEN.json',
            'LRBEN.json',
        ):
            data_pipe_list.extend(_prepare_teochatlas_geochat_eval(data_root, filename))
        return data_pipe_list
    elif benchmark == 'teochatlas_temporal':
        for filename in teochatlas_temporal_files:
            yield from _prepare_teochatlas_eval(data_root, filename, prompt_type, min_pixels, max_pixels)
    else:
        filename = benchmark.replace('teochatlas_', '') + '.json'
        file_mapping = {file.lower(): file for file in teochatlas_temporal_files}
        if filename.lower() not in file_mapping:
            raise ValueError(f"Unknown name: {benchmark}")
        filename = file_mapping[filename.lower()]
        yield from _prepare_teochatlas_eval(data_root, filename, prompt_type, min_pixels, max_pixels)
