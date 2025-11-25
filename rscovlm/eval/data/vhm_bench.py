import os
import json


def _prepare_vhm_eval(data_root, ann_filename):
    data_root = os.path.join(data_root, 'datasets_eval')
    ann_filepath = os.path.join(data_root, ann_filename)

    with open(ann_filepath, 'r') as f:
        data = json.load(f)

    for idx, x in enumerate(data):
        image_name = x['image']
        image_folder = x['image_path']
        image_path = os.path.join(data_root, image_folder, image_name)

        if ann_filename.startswith('cls'):
            x['conversations'][0]['value'] = (
                x['conversations'][0]['value']
                .replace("Answer with one word or short phrase.", "")
                .strip()
            )

        conversations = x['conversations']
        assert len(conversations) == 2 and \
            conversations[0]['from'] == 'human' and \
            conversations[1]['from'] == 'gpt' and \
            '<image>' not in conversations[0]

        question = conversations[0]['value'].strip()
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


def prepare_vhm_eval(data_root, benchmark):
    data_pipe_list = []
    if benchmark == 'vhm_open_ended_qa':  # Table 5 in VHM paper
        for filename in (
            'counting_dota-test_open.json',  # object counting
            'imgType_mcq.json',  # image modality/type
            # 'gsd_dota_fbp.json',  # image resolution
            'obj_meas_dota_test.json',  # geometric measurement
            # 'bfv_crowdai_val.json',  # building vectorizing
            # 'mlc_fbp_test.json', 'mlc_gid_test.json',  # multi-label classification
        ):
            data_pipe_list.extend(_prepare_vhm_eval(data_root, filename))

    elif benchmark == 'vhm_open_ended_qa_full':  # Table 5 in VHM paper
        for filename in (
            'counting_dota-test_open.json',  # object counting
            'imgType_mcq.json',  # image modality/type
            'gsd_dota_fbp.json',  # image resolution
            'obj_meas_dota_test.json',  # geometric measurement
            'bfv_crowdai_val.json',  # building vectorizing
            'mlc_fbp_test.json', 'mlc_gid_test.json',  # multi-label classification
        ):
            data_pipe_list.extend(_prepare_vhm_eval(data_root, filename))

    elif benchmark == 'vhm_scene_cls':  # Table 5 in VHM paper
        for filename in (
            'cls_AID.json',
            "cls_METER_ML.json",
            "cls_NWPU_RESISC45.json",
            "cls_SIRI_WHU.json",
            "cls_WHU_RS19.json"
        ):
            data_pipe_list.extend(_prepare_vhm_eval(data_root, filename))        

    elif benchmark == 'vhm_rsvqa':  # Table 7/8 in VHM paper
        for filename in (
            'RSVQA_HR-comp_RSVQA.json',
            'RSVQA_HR-presence_RSVQA.json',
            'RSVQA_LR-comp_RSVQA.json',
            'RSVQA_LR-presence_RSVQA.json',
            'RSVQA_LR-rural_urban_RSVQA.json'
        ):
            data_pipe_list.extend(_prepare_vhm_eval(data_root, filename))

    elif benchmark == 'vhm_hnstd_qa':  # Table 11 in VHM paper
        for filename in (
            'presence_mo_dota.json',  # presence
            'color_dota-test_fair1m-val_open.json',  # color
            'abspos_dota-test_mc.json',  # absolute position
            'relpos_dota-test_mc.json',  # relative position
        ):
            data_pipe_list.extend(_prepare_vhm_eval(data_root, filename))

    else:
        raise ValueError(f"Unknown name: {benchmark}")
    return data_pipe_list


if __name__ == '__main__':
    data_root = './playground/data/VHM_eval_dataset'
    vhm_open_ended_qa_data = prepare_vhm_eval(data_root, 'vhm_open_ended_qa_clssampled')
    hnst_qa_data = prepare_vhm_eval(data_root, 'vhm_hnstd_qa')
    import ipdb; ipdb.set_trace()
