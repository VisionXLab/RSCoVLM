import json


def extract_examples(data, pattern):
    selected_data, other_data = [], []
    for sample in data:
        if isinstance(pattern, list):
            matched = all(p in json.dumps(sample) for p in pattern)
        else:
            matched = pattern in json.dumps(sample)
        if matched:
            selected_data.append(sample)
        else:
            other_data.append(sample)
    return selected_data, other_data


def main():
    data = json.load(open("playground/data/VHM_dataset_sft/list_sft.json", "r"))

    vqa_data, data = extract_examples(data, "{VQA}")
    for sample in vqa_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{VQA} ", "")
        )
    print(f"VQA: {len(vqa_data)}")

    with open("playground/data/vhm_sft_mix_qwen2_5_vl_conversations.json", "w") as f:
        json.dump(vqa_data, f, indent=4)

    vg_data, data = extract_examples(data, "{VG}")  # thrown
    print(f"VG: thrown ({len(vg_data)})")
    
    cls_data, data = extract_examples(data, "{CLS}")
    for sample in cls_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{CLS} ", "")
            .replace("Answer with one word or short phrase.", "")
            .strip()
        )
    print(f"CLS: {len(cls_data)}")

    count_data, data = extract_examples(data, ["{IT}", "answer the number"])
    for sample in count_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{IT} ", "")
            .strip()
        )
    print(f"COUNT: {len(count_data)}")

    modality_mcq_data, data = extract_examples(data, "{IT} What type of image is this?")
    for sample in modality_mcq_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{IT} ", "")
            .replace("Pealse", "Please")
            .strip()
        )
        sample['conversations'][1]['value'] = (
            sample['conversations'][1]['value']
            .replace("A.  ", "A.").replace("B.  ", "B.").replace("C.  ", "C.").replace("D.  ", "D.")
            .strip()
        )
    print(f"MODALITY CHOOSING: {len(modality_mcq_data)}")

    resolution_reg_data, data = extract_examples(data, "{IT} What is the spatial resolution")  # thrown
    print(f"RESOLUTION REGRESSION: thrown ({len(resolution_reg_data)})")

    geometric_measurement_data, data = extract_examples(data, ["{IT}", "actual length"])
    for sample in geometric_measurement_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{IT} ", "")
            .strip()
        )
    print(f"GEOMETRIC MEASUREMENT: {len(geometric_measurement_data)}")

    building_vectorizing_data, data = extract_examples(data, "{IT} Please output the outlines of buildings")  # thrown
    print(f"BUILDING VECTORIZING: thrown ({len(building_vectorizing_data)})")

    multi_label_cls_data, data = extract_examples(data, 'deepglobe_lc')
    for sample in multi_label_cls_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{IT} ", "")
            .strip()
        )
    print(f"MULTI-LABEL CLASSIFICATION: {len(multi_label_cls_data)}")

    invisible_data, data = extract_examples(data, ["{IT}", "not visible"])
    for sample in invisible_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{IT} ", "")
            .strip()
        )
    print(f"INVISIBLE: {len(invisible_data)}")

    hnstd_data, data = extract_examples(data, "{IDK}")
    for sample in hnstd_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("{IDK} ", "")
            .strip()
        )
        sample['conversations'][1]['value'] = (
            sample['conversations'][1]['value']
            .strip()
        )
    print(f"HNSDT: {len(hnstd_data)}")

    versad_instruct_data = data
    for sample in versad_instruct_data:
        sample['conversations'][0]['value'] = (
            sample['conversations'][0]['value']
            .replace("<image>\n ", "<image>\n")
            .strip()
        )
        for idx in range(1, len(sample['conversations'])):
            sample['conversations'][idx]['value'] = (
                sample['conversations'][idx]['value']
                .strip()
            )
    print(f"VERSAD INSTRUCTION: {len(versad_instruct_data)}")

    invisible_versad_instruct_data = []
    for sample in versad_instruct_data:
        if "not visible" in json.dumps(sample):
            invisible_versad_instruct_data.append(sample)
    print(f"INVISIBLE examples in VERSAD INSTRUCTION: {len(invisible_versad_instruct_data)}")

    new_data = vqa_data + cls_data + count_data + modality_mcq_data + geometric_measurement_data + multi_label_cls_data + hnstd_data + versad_instruct_data
    print(f"NEW DATA: {len(new_data)}")

    with open("playground/data/vhm_sft_mix_qwen2_5_vl_conversations.json", "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == '__main__':
    main()
