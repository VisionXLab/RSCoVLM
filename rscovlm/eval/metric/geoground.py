from .metric import extract_qwen25vl_bbox_answer, resize_bbox, iou


def _compute_geoground_acc(data_pipe_list, prompt_type='plain'):
    correct_number = 0
    for sample in data_pipe_list:
        ground_truth = sample['bbox']
        
        model_answer, normalized = extract_qwen25vl_bbox_answer(sample['output'], prompt_type)
        assert not normalized, "Normalized bbox is not implemented yet"
        model_answer = resize_bbox(model_answer, sample['original_width'], sample['original_height'], sample['resized_width'], sample['resized_height'])
        
        correct = 0
        if model_answer is not None:
            if iou(model_answer, ground_truth) > 0.5:
                correct = 1
        correct_number += correct

    accuracy = correct_number / len(data_pipe_list) * 100
    return accuracy


def compute_geoground_acc(data_pipe_list, prompt_type='plain'):
    dataset_name_to_data_pipe_list = {}
    for sample in data_pipe_list:
        if sample['dataset_name'] not in dataset_name_to_data_pipe_list:
            dataset_name_to_data_pipe_list[sample['dataset_name']] = []
        dataset_name_to_data_pipe_list[sample['dataset_name']].append(sample)

    dataset_name_to_accuracy = {}
    for dataset_name, data_pipe_list in dataset_name_to_data_pipe_list.items():
        accuracy = _compute_geoground_acc(data_pipe_list, prompt_type)
        dataset_name_to_accuracy[dataset_name] = accuracy
    return dataset_name_to_accuracy
