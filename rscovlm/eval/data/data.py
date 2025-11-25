import os
import json
import logging
import xmltodict

logger = logging.getLogger(__name__)


def load_json_or_jsonl(*args):
    data_list = []
    ext = os.path.splitext(args[-1])[-1]
    if ext == '.jsonl':
        with open(os.path.join(*args), "r", encoding="utf-8") as f:
            for line in f:
                data_list.append(json.loads(line))
    elif ext == ".json":
        data_list = json.load(open(os.path.join(*args), "r"))
    else:
        raise ValueError(f"Unsupported file type: {ext} (only .json and .jsonl are supported)")
    return data_list


def get_incontext_msg(benchmark, icl_shot, ppl_json_path, unique=False):
    incontext_msg = []
    if icl_shot == 0:
        return incontext_msg
    ppl_json_filename = f"ppl_{benchmark}_train.json"
    incontext_items = json.load(open(os.path.join(ppl_json_path, ppl_json_filename), "r"))
    # "image_id", "gt_idx", "ppl", "msg_response", "msg_instruction"

    if not unique:
        selected_items = incontext_items[:icl_shot]
    else:
        selected_items = []
        seen_responses = set()
        remaining_items = []

        for item in incontext_items:
            response_content = item["msg_response"][0]["content"].strip().lower()
            if response_content not in seen_responses:
                if len(selected_items) < icl_shot:
                    selected_items.append(item)
                    seen_responses.add(response_content)
                else:
                    break
            else:
                remaining_items.append(item)
        
        if len(selected_items) < icl_shot:
            needed = icl_shot - len(selected_items)
            selected_items.extend(remaining_items[:needed])

    for ic in selected_items:
        incontext_msg.extend(ic["msg_instruction"])
        incontext_msg.extend(ic["msg_response"])
    return incontext_msg


def check_text_mode(data, mode): # plain text mode  # TODO: what is this?
    isjson = "json" in data["msg_instruction"][0]["content"][1]["text"].lower()
    if isjson and mode:
        raise ValueError("Set --grounding_prompt_type to `json` for matching the PPL File")
    elif not isjson and not mode:
        raise ValueError("Set --grounding_prompt_type to `plain` for matching the PPL File")
    return


def prepare_vlm_r1_rec(benchmark, data_root, image_root, grounding_prompt_type):
    logger.info(f"Processing {benchmark}...") # ds=benchmark
    ds_path = os.path.join(data_root, f"{benchmark}.json")
    data = json.load(open(ds_path, "r"))

    for idx, x in enumerate(data):
        image_path = os.path.join(image_root, x['image'])
        
        if grounding_prompt_type == 'json':
            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"Locate the {x['normal_caption']}, output the bbox coordinates using JSON format"}
                ]}
            ]
        elif grounding_prompt_type == 'plain':
            message = [
                {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"find the {x['normal_caption']}"}
                ]}
            ]
        else:
            raise ValueError(f"Unknown grounding prompt type: {grounding_prompt_type}")

        x["message"] = message
        x["idx"] = idx
    return data


def prepare_rs_caption(image_root, data_json_root, benchmark, icl_shot, ppl_json_path):
    incontext_msg = get_incontext_msg(benchmark, icl_shot, ppl_json_path)
    test_data = json.load(open(os.path.join(data_json_root, f'{benchmark}_test.json'), "r"))
    for idx, x in enumerate(test_data):
        image_path = os.path.join(image_root, x['image_id'])
        question = x['question']
        message = incontext_msg + [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]}
        ]
        x["message"] = message
        x["idx"] = idx

    return test_data


def prepare_rrsisd_res(rrsisd_root, ann_type='refer'):
    sys_prompt = "As an AI assistant, you specialize in segmentation, delivering segmentation results as Row-wise RLE format. Each row is represented as '<value>*<count>', and rows are separated by ';'. The last row ends with a semicolon. For example, '0*1,1*2;0*3;' represents the mask."  # TODO: refine this prompt and unify with training
    prompt_template = "Locate the {prompt}, output the segmentation mask"
    # prompt_template = "Please provide the segmentation mask of the region this sentence describes: {prompt}"
    image_folder = os.path.join(rrsisd_root, "images/rrsisd/JPEGImages/")

    samples = []
    if ann_type.upper() == 'REFER':
        from rscovlm.utils import REFER
        refer_api = REFER(rrsisd_root, "rrsisd", "unc")
        ref_ids = sorted(refer_api.getRefIds(split='test'))

        for ref_id in ref_ids:
            ref = refer_api.Refs[ref_id]
            refer_txt = ref['sentences'][0]['sent']
            image_file_path = os.path.join(image_folder, ref['file_name'])
            ref['refer_txt'] = refer_txt
            ref['image_file_path'] = image_file_path
            samples.append(ref)
        print(f"Loaded {len(samples)} samples.")

    elif ann_type.lower() == 'xml':
        xml_folder = os.path.join(rrsisd_root, 'images/rrsisd/ann_split')
        xml_file_list = list(os.listdir(xml_folder))
        num_test_xml = 0
        for xml_file in tqdm(sorted(xml_file_list)):
            with open(os.path.join(xml_folder, xml_file), 'r') as f:
                ref = xmltodict.parse(f.read())['annotation']
            if ref['split'] != 'test':
                continue
            ref["xml_file_path"] = os.path.join(xml_folder, xml_file)
            num_test_xml += 1
            image_file_path = os.path.join(image_folder, ref['filename'])
            object = ref.pop('object')
            if not isinstance(object, list):
                object = [object]
            for obj in object:
                refer_txt = obj['description']
                samples.append({
                    'image_file_path': image_file_path,
                    'refer_txt': refer_txt,
                    **ref, **obj
                })
        print(f"Loaded {len(samples)} samples from {len(xml_file_list)} XML files (num_test_xml={num_test_xml}).")
    
    else:
        raise ValueError("Unknown annotation type")
        
    for idx, x in enumerate(samples):
        image_file_path = x['image_file_path']
        refer_txt = x['refer_txt']

        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_file_path}"},
                {"type": "text", "text": prompt_template.format(prompt=refer_txt)}
            ]},
        ]

        x["message"] = message
        x["idx"] = idx

    return samples
