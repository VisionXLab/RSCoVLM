import os
import re
import json
import warnings
import simplejson
from tqdm import tqdm
from rapidfuzz import process  # pip install rapidfuzz

import torch
import mmcv
from mmengine import dump as mmengine_dump
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmrotate.visualization import RotLocalVisualizer
from mmrotate.structures import QuadriBoxes, RotatedBoxes

from ..data.dense_det import build_eval_dataset


def get_evaluator(dataset_type, is_test_set, results_path=None, box_type='rbox'):
    def set_split_attr(evaluator, split_attr):
        evaluator.split = split_attr
        return evaluator

    def _get_submission_evaluator(METRIC_CLASS, _results_path):
        return set_split_attr(METRIC_CLASS(
            format_only=True,
            predict_box_type=box_type,
            merge_patches=True,
            outfile_prefix=f'{_results_path}/dota_Task1'
        ), "test")

    if dataset_type == "fair1m":
        from lmmrotate.modules.fair_metric import FAIRMetric
        if is_test_set:
            return _get_submission_evaluator(FAIRMetric, results_path)
        else:
            return set_split_attr(FAIRMetric(metric="mAP", predict_box_type=box_type), "train")
    elif dataset_type == "fair1m_2.0_train":
        from lmmrotate.modules.fair_metric import FAIRMetric
        return set_split_attr(FAIRMetric(metric="mAP", predict_box_type=box_type), "val" if is_test_set else "train")
    elif dataset_type in ["dota", "dota_512"]:
        from mmrotate.evaluation import DOTAMetric
        if is_test_set:
            return _get_submission_evaluator(DOTAMetric, results_path)
        else:
            return set_split_attr(DOTAMetric(metric="mAP", predict_box_type=box_type), "trainval")
    elif dataset_type == "dota_train":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP", predict_box_type=box_type), "val" if is_test_set else "train")
    elif dataset_type == "dior":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP", predict_box_type=box_type), "test" if is_test_set else "trainval")
    elif dataset_type == "srsdd":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP", predict_box_type=box_type), "test" if is_test_set else "train")
    elif dataset_type == "rsar":
        from mmrotate.evaluation import DOTAMetric
        return set_split_attr(DOTAMetric(metric="mAP", predict_box_type=box_type), "test" if is_test_set else "trainval")
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")


def parse_plain_output(
    output: str, 
    cls_map: dict, 
    original_width: int, 
    original_height: int, 
    resized_width: int, 
    resized_height: int
) -> tuple[list, list, list]:
    output = output.strip()

    all_polygons = []
    all_scores = []
    all_labels = []
    for line in output.splitlines():
        if len(line.strip()) == 0:
            continue
            
        try:
            poly, label = line.split(' ', 1)
            poly = poly.split(',')
            poly = [int(coord) for coord in poly]
        except Exception as e:
            warnings.warn(f"Instance with wrong line format:\n\n{line}\n\n")
            continue
        label = label.strip().lower()

        if len(poly) != 8:
            warnings.warn(f"Instance with wrong polygon format:\n\n{poly}\n\n")
            if len(poly) > 8:
                poly = poly[:8]
            else:
                # import ipdb; ipdb.set_trace()
                continue

        if label in cls_map:
            label = cls_map[label]
        else:
            fuzzy_matched_cat = process.extractOne(label, cls_map.keys())[0]
            warnings.warn(f"Fuzzy matched {label} to {fuzzy_matched_cat}\n\n")
            label = cls_map[fuzzy_matched_cat]

        for i in range(0, len(poly), 2):
            poly[i] = poly[i] * original_width / resized_width
            poly[i + 1] = poly[i + 1] * original_height / resized_height

        all_polygons.append(poly)
        all_scores.append(1.0)
        all_labels.append(label)
    return all_polygons, all_scores, all_labels


def parse_json_output(
    output: str, 
    cls_map: dict, 
    original_width: int, 
    original_height: int, 
    resized_width: int, 
    resized_height: int
) -> tuple[list, list, list]:
    output = output.replace("```json\n", "").replace("\n```", "")
    try:
        output = simplejson.loads(output)
    except Exception as e:
        # try to resolve usual conner cases
        # 1. the `[]` may be incorrect `()`
        output = output.replace('(', '[').replace(')', ']')
        # 2. there may be multiple continuous `,`
        output = re.sub(r'(, *)+', ', ', output)
        # 3. supplement the missed `,` between two numbers
        output = re.sub(r'(\d)\s+(\d)', r'\1, \2', output)
        # 4. there may be multiple continuous `"label": `
        output = re.sub(r'("label":\s*)+', '"label": ', output)
        # # 5. there may be unfinished item before (6 can resolve this)
        # last_brace_pos = output.rfind('}')
        # if not (last_brace_pos == -1 or output.endswith("}") or output.endswith("}\n")):
        #     output = output[:last_brace_pos + 2]
        #     output = output.strip(',') + "]"
        # 6. a long response may contain a small mistake
        lines = output.splitlines()
        new_lines = []
        for line in lines:
            try:
                simplejson.loads(line.strip().strip(','))
            except Exception as e:
                continue
            new_lines.append(line)
        output = '[\n' + '\n'.join(new_lines).rstrip(', ') + '\n]'

        try:
            output = simplejson.loads(output)
        except Exception as e2:
            warnings.warn(f"Failed to parse output partially:\n\n{e2}\n\n{output}\n\n")
            import ipdb; ipdb.set_trace()
            output = []

    all_polygons = []
    all_scores = []
    all_labels = []
    for inst in output:
        if "quadrant bbox" not in inst or "label" not in inst:
            warnings.warn(f"Instance with wrong format:\n\n{output}\n\n")
            # import ipdb; ipdb.set_trace()
            continue

        poly = inst["quadrant bbox"]
        label = inst["label"].lower()

        if not (isinstance(poly, list) and len(poly) == 8 and all(isinstance(n, int) for n in poly)):
            warnings.warn(f"Instance with wrong polygon format:\n\n{poly}\n\n")
            if (isinstance(poly, list) and len(poly) > 8 and all(isinstance(n, int) for n in poly)):
                poly = poly[:8]
            else:
                # import ipdb; ipdb.set_trace()
                continue

        if label in cls_map:
            label = cls_map[label]
        else:
            fuzzy_matched_cat = process.extractOne(label, cls_map.keys())[0]
            warnings.warn(f"Fuzzy matched {label} to {fuzzy_matched_cat}\n\n")
            label = cls_map[fuzzy_matched_cat]

        for i in range(0, len(poly), 2):
            poly[i] = poly[i] * original_width / resized_width
            poly[i + 1] = poly[i + 1] * original_height / resized_height

        all_polygons.append(poly)
        all_scores.append(1.0)
        all_labels.append(label)
    return all_polygons, all_scores, all_labels


EMPTY_JSON_OUTPUT = json.dumps([])
EMPTY_PLAIN_OUTPUT = ''
EMPTY_OUTPUT = {
    'json': EMPTY_JSON_OUTPUT,
    'plain': EMPTY_PLAIN_OUTPUT,
}


def parse_output(data_pipe, cls_map, box_type='rbox', prompt_type='json', clear_pred_for_empty_gt=False):
    output = data_pipe['output']

    original_width = data_pipe['original_width']
    original_height = data_pipe['original_height']
    resized_width = data_pipe['resized_width']
    resized_height = data_pipe['resized_height']

    if clear_pred_for_empty_gt:
        if len(data_pipe['gt_instances']['bboxes']) == 0:
            output = EMPTY_OUTPUT[prompt_type]

    if output.lower().strip().rstrip('.') == 'there are none':
        output = EMPTY_OUTPUT[prompt_type]

    if prompt_type == 'json':
        all_polygons, all_scores, all_labels = parse_json_output(output, cls_map, original_width, original_height, resized_width, resized_height)
    elif prompt_type == 'plain':
        all_polygons, all_scores, all_labels = parse_plain_output(output, cls_map, original_width, original_height, resized_width, resized_height)
    else:
        raise NotImplementedError(f"Unknown prompt type: {prompt_type}")
        
    bboxes = QuadriBoxes(all_polygons)

    if box_type == 'rbox':
        bboxes = bboxes.convert_to('rbox')
    else:
        assert box_type == 'qbox', f"Unknown box type: {box_type}"

    if len(all_polygons) == 0:
        bboxes = bboxes.empty_boxes()

    pred_instances = {
        'bboxes': bboxes.tensor, 
        'labels': torch.as_tensor(all_labels), 
        'scores': torch.as_tensor(all_scores),
    }
    return pred_instances


def override_gt(data_pipe, data_sample):
    gt_instances = data_sample.gt_instances
    if hasattr(gt_instances, 'bboxes'):
        data_pipe['gt_instances']['bboxes'] = gt_instances.bboxes.tensor
    if hasattr(gt_instances, 'labels'):
        data_pipe['gt_instances']['labels'] = gt_instances.labels
    ignored_instances = data_sample.ignored_instances
    if hasattr(ignored_instances, 'bboxes'):
        data_pipe['ignored_instances']['bboxes'] = ignored_instances.bboxes.tensor
    if hasattr(ignored_instances, 'labels'):
        data_pipe['ignored_instances']['labels'] = ignored_instances.labels


def samplelist_listdata2tensor(samplelist, box_dim=8):
    for data_pipe in samplelist:
        if 'labels' in data_pipe['gt_instances']:
            gt_instances_labels = data_pipe['gt_instances']['labels']
            if not isinstance(gt_instances_labels, torch.Tensor):
                if len(gt_instances_labels) == 0:
                    data_pipe['gt_instances']['labels'] = torch.empty(0, dtype=torch.int64)
                else:
                    data_pipe['gt_instances']['labels'] = torch.as_tensor(gt_instances_labels)
        if 'bboxes' in data_pipe['gt_instances']:
            gt_instances_bboxes = data_pipe['gt_instances']['bboxes']
            if not isinstance(gt_instances_bboxes, torch.Tensor):
                if len(gt_instances_bboxes) == 0:
                    data_pipe['gt_instances']['bboxes'] = torch.empty(0, box_dim, dtype=torch.float32)
                else:
                    data_pipe['gt_instances']['bboxes'] = torch.as_tensor(gt_instances_bboxes)
        if 'labels' in data_pipe['ignored_instances']:
            ignored_instances_labels = data_pipe['ignored_instances']['labels']
            if not isinstance(ignored_instances_labels, torch.Tensor):
                if len(ignored_instances_labels) == 0:
                    data_pipe['ignored_instances']['labels'] = torch.empty(0, dtype=torch.int64)
                else:
                    data_pipe['ignored_instances']['labels'] = torch.as_tensor(ignored_instances_labels)
        if 'bboxes' in data_pipe['ignored_instances']:
            ignored_instances_bboxes = data_pipe['ignored_instances']['bboxes']
            if not isinstance(ignored_instances_bboxes, torch.Tensor):
                if len(ignored_instances_bboxes) == 0:
                    data_pipe['ignored_instances']['bboxes'] = torch.empty(0, box_dim, dtype=torch.float32)
                else:
                    data_pipe['ignored_instances']['bboxes'] = torch.as_tensor(ignored_instances_bboxes)
        if 'pred_instances' in data_pipe:
            pred_instances_labels = data_pipe['pred_instances']['labels']
            if not isinstance(pred_instances_labels, torch.Tensor):
                if len(pred_instances_labels) == 0:
                    data_pipe['pred_instances']['labels'] = torch.empty(0, dtype=torch.int64)
                else:
                    data_pipe['pred_instances']['labels'] = torch.as_tensor(pred_instances_labels)
            pred_instances_bboxes = data_pipe['pred_instances']['bboxes']
            if not isinstance(pred_instances_bboxes, torch.Tensor):
                if len(pred_instances_bboxes) == 0:
                    data_pipe['pred_instances']['bboxes'] = torch.empty(0, box_dim, dtype=torch.float32)
                else:
                    data_pipe['pred_instances']['bboxes'] = torch.as_tensor(pred_instances_bboxes)
            pred_instances_scores = data_pipe['pred_instances']['scores']
            if not isinstance(pred_instances_scores, torch.Tensor):
                if len(pred_instances_scores) == 0:
                    data_pipe['pred_instances']['scores'] = torch.empty(0, dtype=torch.float32)
                else:
                    data_pipe['pred_instances']['scores'] = torch.as_tensor(pred_instances_scores)


def make_sure_gt_instances_are_qbox(data_pipe_list):
    for data_pipe in data_pipe_list:
        gt_instances_bboxes = data_pipe['gt_instances']['bboxes']
        if gt_instances_bboxes.shape[1] == 5:
            data_pipe['gt_instances']['bboxes'] = RotatedBoxes(gt_instances_bboxes).convert_to('qbox').tensor
        ignored_instances_bboxes = data_pipe['ignored_instances']['bboxes']
        if ignored_instances_bboxes.shape[1] == 5:
            data_pipe['ignored_instances']['bboxes'] = RotatedBoxes(ignored_instances_bboxes).convert_to('qbox').tensor


def datapipe2datasample(data_pipe, box_dim=8):
    samplelist_listdata2tensor([data_pipe], box_dim=box_dim)

    gt_instances = data_pipe['gt_instances']
    ignored_instances = data_pipe['ignored_instances']
    pred_instances = data_pipe['pred_instances']
    image_meta = {
        k:v for k, v in data_pipe.items() 
        if k not in ('gt_instances', 'ignored_instances', 'pred_instances', 'output', 'message')
    }

    data_sample = DetDataSample(metainfo=image_meta)

    pred_instance_data = InstanceData()
    pred_instance_data.bboxes = pred_instances['bboxes']
    pred_instance_data.labels = pred_instances['labels']
    pred_instance_data.scores = pred_instances['scores']
    data_sample.pred_instances = pred_instance_data

    if not (isinstance(ignored_instances, dict) and len(ignored_instances) == 0):
        ignore_instance_data = InstanceData()
        ignore_instance_data.bboxes = ignored_instances['bboxes']
        ignore_instance_data.labels = ignored_instances['labels']
        data_sample.ignored_instances = ignore_instance_data

    if not (isinstance(gt_instances, dict) and len(gt_instances) == 0):
        gt_instance_data = InstanceData()
        gt_instance_data.bboxes = gt_instances['bboxes']
        gt_instance_data.labels = gt_instances['labels']
        data_sample.gt_instances = gt_instance_data
    return data_sample


def eval_dense_det(
    data_pipe_list, 
    benchmark, 
    output_folder, 
    prompt_type='json',
    eval_box_type='rbox', 
    visualize_num=False, 
    save_map_for_each_example=None, 
    clear_pred_for_empty_gt=False,
):
    results_path = os.path.join(output_folder, benchmark, f"res_dense_det_{benchmark}")
    dataset = build_eval_dataset(benchmark, box_type=eval_box_type)
    name2idx = {datasample.file_name: idx for idx, (_, datasample) in enumerate(dataset)}

    if visualize_num > 0:
        visualizer = RotLocalVisualizer(name='visualizer', vis_backends=[dict(type='LocalVisBackend')])
        visualizer.dataset_meta = dataset.metainfo

    evaluator = get_evaluator(dataset.dataset_type, dataset.is_test_set, results_path, box_type=eval_box_type)
    evaluator.dataset_meta = dataset.metainfo

    for idx, data_pipe in enumerate(tqdm(data_pipe_list, desc=f"Evaluating {benchmark}")):
        # get prediction instances
        data_pipe['pred_instances'] = parse_output(
            data_pipe, dataset.cls_map, eval_box_type, prompt_type, clear_pred_for_empty_gt)

        # get gt instances 
        # (sometimes optional: there have been gt instances in the data pipe, but the box type 
        # may be mismatched, here we just override the gt instances with loaded data_sample)
        data_idx = name2idx[data_pipe['file_name']]
        override_gt(data_pipe, dataset[data_idx][1])

        samplelist_listdata2tensor([data_pipe], box_dim={'qbox': 8, 'rbox': 5}[eval_box_type])  # required, boxes are saved as list data in json, but the evaluator requires tensor type
        # make_sure_gt_instances_are_qbox([data_pipe])  # optional, it is just for hardcode for qbox type
        evaluator.process(data_batch=None, data_samples=[data_pipe])

        if visualize_num > 0:
            # if len(data_pipe['gt_instances']['bboxes']) == 0:  # optional, ignore cases with empty gt but false positive predictions
            #     continue
            os.makedirs(os.path.join(output_folder, 'vis'), exist_ok=True)
            img = mmcv.imread(data_pipe['img_path'])
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            visualizer.add_datasample(
                'result',
                img,
                data_sample=datapipe2datasample(data_pipe),
                out_file=os.path.join(output_folder, 'vis', data_pipe['file_name']),
                pred_score_thr=0)
            visualize_num -= 1
            # if visualize_num == 0:  # optional, only calculate the scores for the visualized images
            #     break
            import ipdb; ipdb.set_trace()

        if save_map_for_each_example is not None and len(data_pipe['gt_instances']['bboxes']) == 0:
            if not hasattr(evaluator, 'results_backup'):
                evaluator.results_backup = []
                data_pipe['map_sample'] = evaluator.compute_metrics(evaluator.results)['AP50']
                import ipdb; ipdb.set_trace()
            evaluator.results_backup.extend(evaluator.results)
            evaluator.results = []

    if save_map_for_each_example is not None:
        evaluator.results = evaluator.results_backup
        with open(save_map_for_each_example, 'w') as f:
            json.dump(data_pipe_list, f, indent=4)

    # save pickle results
    pickle_results_path = os.path.join(output_folder, benchmark, f"pickle_results.pkl")
    mmengine_dump(data_pipe_list, pickle_results_path)
    return evaluator.compute_metrics(evaluator.results)


def transform_rbox2qbox_for_gt_in_inference_results(data_pipe_list, benchmark):
    dataset = build_eval_dataset(benchmark)
    name2idx = {datasample.file_name: idx for idx, (_, datasample) in enumerate(dataset)}

    for data_pipe in data_pipe_list:
        file_name = data_pipe['file_name']
        sample_idx = name2idx[file_name]
        data_sample = dataset[sample_idx][1]

        data_pipe['gt_instances']['bboxes'] = data_sample.gt_instances.bboxes.tensor
        data_pipe['gt_instances']['labels'] = data_sample.gt_instances.labels
        data_pipe['ignored_instances']['bboxes'] = data_sample.ignored_instances.bboxes.tensor
        data_pipe['ignored_instances']['labels'] = data_sample.ignored_instances.labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate dense detection results')
    parser.add_argument('inference_results_path', type=str, help='Json path to the data pipe list with inference results')
    parser.add_argument('benchmark', type=str, help='Benchmark name')
    parser.add_argument('--vis_num', default=-1, type=int, help='Number of images to visualize')
    parser.add_argument('--prompt_type', default='plain', type=str, help='Prompt type')
    parser.add_argument('--eval_box_type', default='rbox', type=str, help='Box type')
    parser.add_argument('--save_map_for_each_example', type=str, default=None, help='Calculate mAP for each example')
    parser.add_argument('--clear_pred_for_empty_gt', action='store_true', default=False, help='Clear prediction for empty gt')
    args = parser.parse_args()

    args.save_path = os.path.dirname(args.inference_results_path)

    with open(args.inference_results_path, 'r') as f:
        data_pipe_list = json.load(f)

    eval_dense_det(
        data_pipe_list, 
        args.benchmark, 
        args.save_path, 
        args.prompt_type,
        args.eval_box_type, 
        args.vis_num, 
        args.save_map_for_each_example,
        args.clear_pred_for_empty_gt
    )
