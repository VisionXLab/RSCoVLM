import os
from .data import load_json_or_jsonl
from rscovlm.training.data.visual_cot_dataset import SYSTEM_PROMPT as VISUAL_COT_SYSTEM_PROMPT


def get_qa_message(image_path, question, visual_cot: bool = False):
    if visual_cot:
        message = [
            {"role": "system", "content": VISUAL_COT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": f"{question}\nZoom in to the region of interest and answer the question."}  # TODO: do we require `answer the question using a single word or pharse`
            ]},
        ]
    else:
        message = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]}
        ]
    return message


def prepare_lrsvqa(jsonl_path, image_root, visual_cot: bool = False):
    data = load_json_or_jsonl(jsonl_path)
    for idx, x in enumerate(data):

        x["message"] = get_qa_message(
            image_path=os.path.join(image_root, x['image']),
            question=x['text'], 
            visual_cot=visual_cot and x['category'] not in ['count']
        )

        x["idx"] = idx

    data.sort(key=lambda x: (-len(x['message']), x['idx']))
    return data


def prepare_mme_realworld_remote_sensing(data_root, visual_cot: bool = False):
    prompt = 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.'
    ann_path = "MME_RealWorld_remote_sensing.json"
    data = load_json_or_jsonl(data_root, ann_path)
    for idx, x in enumerate(data):
        choice_prompt = ' The choices are listed below: \n'
        for choice in x['Answer choices']:
            choice_prompt += choice + "\n"
        
        x["message"] = get_qa_message(
            image_path=os.path.join(data_root, x['Image']), 
            question=x['Text'] + choice_prompt + prompt + '\nThe best answer is:',
            visual_cot=visual_cot and x['Category'] not in ['count', 'position']
        )

        x["idx"] = idx
        x["ground_truth"] = x.pop("Ground truth")
    data.sort(key=lambda x: (-len(x['message']), x['idx']))
    return data
