import re
import json
import logging
from pycocoevalcap.cider.cider import Cider
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import defaultdict
import torch
from torchvision.ops.boxes import box_area
from shapely.geometry import Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def compute_identical_acc(data_pipe_list):
    correct = 0
    
    for item in data_pipe_list:
        gt = item["ground_truth"].lower().strip().strip(".")
        pred = item["output"].lower().strip().strip(".")
        
        if pred == gt:
            correct += 1
    
    if len(data_pipe_list) == 0:
        return None
    else:
        acc = correct / len(data_pipe_list)
        return {"acc": acc}


def compute_contain_acc(data_pipe_list):
    correct = 0
    
    for item in data_pipe_list:
        gt = item["ground_truth"].lower().strip().strip(".")
        pred = item["output"].lower().strip().strip(".")
        
        if gt in pred:
            correct += 1
    
    if len(data_pipe_list) == 0:
        return None
    else:
        acc = correct / len(data_pipe_list)
        return {"acc": acc}


def compute_startswith_acc(data_pipe_list):
    correct = 0
    
    for item in data_pipe_list:
        gt = item["ground_truth"].lower().strip().strip(".")
        pred = item["output"].lower().strip().strip(".")
        
        if gt.startswith(pred):
            correct += 1
    
    if len(data_pipe_list) == 0:
        return None
    else:
        acc = correct / len(data_pipe_list)
        return {"acc": acc}


def resize_bbox(bbox, original_width, original_height, resized_width, resized_height):
    x1, y1, x2, y2 = bbox
    x1 = x1 / resized_width * original_width
    x2 = x2 / resized_width * original_width
    y1 = y1 / resized_height * original_height
    y2 = y2 / resized_height * original_height
    return [x1, y1, x2, y2]


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union



def extract_qwen25vl_bbox_answer(content, prompt_type='json'):
    if prompt_type == 'json':
        bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
        # bbox_pattern = r'\[(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\]'
        bbox_match = re.search(bbox_pattern, content)

        if bbox_match:
            bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
            x1, y1, x2, y2 = bbox
            if all(bbox[i] <= 1 for i in range(4)):
                bbox = [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]
                return bbox, True
            return bbox, False
    elif prompt_type == 'plain':
        pattern = r"(\d+),(\d+),(\d+),(\d+)\s+(.+?)\s*$"

        bbox_match = re.match(pattern, content.strip().split('\n')[0].strip())
        if bbox_match:
            x1, y1, x2, y2, _ = bbox_match.groups()
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            if all(bbox[i] <= 1 for i in range(4)):
                bbox = [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]
                return bbox, True
            return bbox, False
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return [0, 0, 0, 0], False


def compute_qwen25vl_vlm_r1_rec_acc(data_pipe_list, prompt_type='json'):
    correct_number = 0
    for sample in data_pipe_list:
        ground_truth = sample['solution']
        ground_truth_normalized = sample['normalized_solution']
        model_answer, normalized = extract_qwen25vl_bbox_answer(sample['output'], prompt_type)
        model_answer = resize_bbox(model_answer, sample['original_width'], sample['original_height'], sample['resized_width'], sample['resized_height'])
        
        correct = 0
        if model_answer is not None:
            if not normalized and iou(model_answer, ground_truth) > 0.5:
                correct = 1
            elif normalized and iou(model_answer, ground_truth_normalized) > 0.5:
                correct = 1
        correct_number += correct

    accuracy = correct_number / len(data_pipe_list) * 100
    return {"acc@0.5": accuracy}


def compute_cider(data_pipe_list):
    gt_captions = {}
    pred_captions = {}
    for i, sample in enumerate(data_pipe_list):
        gt_captions[str(i)] = [gt.lower() for gt in sample['ground_truth']] 
        pred_captions[str(i)] = [sample['output'].lower()] 
    Cider_scorer = Cider()
    cider_score, _ = Cider_scorer.compute_score(gt_captions, pred_captions)
    return {"CIDEr": cider_score * 100}

def rescale_box(bbox, w_rescale, h_rescale):
    # # .view(-1, 2).tolist()
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        coords = [[x1*w_rescale, y1*h_rescale], [x2*w_rescale, y1*h_rescale], [x2*w_rescale, y2*h_rescale], [x1*w_rescale, y2*h_rescale]]
    elif len(bbox) == 8:
        points = [[bbox[i]*w_rescale, bbox[i+1]*h_rescale] for i in range(0, 8, 2)]
        center = np.mean(points, axis=0)
        # clockwise
        angles = np.arctan2(
            np.array([p[1] - center[1] for p in points]),
            np.array([p[0] - center[0] for p in points])
        )
        coords = [points[i] for i in np.argsort(-angles)]
    else:
        raise ValueError("len(bbox) must be 4 or 8")
    return coords

def polygon_giou(poly_pred, poly_gt, w_rescale, h_rescale):

    poly_pred = rescale_box(poly_pred, w_rescale, h_rescale)
    poly_gt = rescale_box(poly_gt, w_rescale, h_rescale)
    p_pred = Polygon(poly_pred)
    p_gt = Polygon(poly_gt)
    if not p_pred.is_valid or not p_gt.is_valid:
        return 0.0

    inter = p_pred.intersection(p_gt).area
    union = p_pred.union(p_gt).area
    if union == 0:
        return 0.0
    # iou
    iou = inter / union
    # 
    all_coords = np.concatenate([poly_pred, poly_gt], axis=0)
    enclosing_poly = Polygon(all_coords).convex_hull
    enclosing_area = enclosing_poly.area

    giou = iou - (enclosing_area - union) / enclosing_area
    return giou

def grec_box_iou_poly(polygons_pred, polygons_gt, w_rescale, h_rescale):
    """Compute [N, M] IoU matrix between two lists of polygons"""

    giou_matrix = torch.zeros((polygons_pred.shape[0], polygons_gt.shape[0]), dtype=torch.float32) # [N, M]
    for i, poly_pred in enumerate(polygons_pred):
        for j, poly_gt in enumerate(polygons_gt):
            giou_matrix[i, j] = polygon_giou(poly_pred, poly_gt, w_rescale, h_rescale)
    return giou_matrix

def adaptive_flatten(tensor: torch.Tensor) -> torch.Tensor:
    """Adaptively flattens the tensor based on its dimensionality:
    - If dim=3: Flattens the last two dimensions (keeps dim0)
    - If dim=2: Returns unchanged
    - Otherwise: Raises ValueError with English message
    """
    if tensor.dim() == 3:
        # Flatten last two dims: [dim0, dim1, dim2] -> [dim0, dim1*dim2]
        return tensor.view(tensor.size(0), -1)
    elif tensor.dim() == 2:
        # Return 2D tensors unchanged
        return tensor
    else:
        raise ValueError(
            f"Input tensor must be 2D or 3D, but got {tensor.dim()}D tensor "
            f"with shape {tuple(tensor.shape)}"
        )

def get_grec_stats_single_sample(stats, pred_boxes_tensor, gt_boxes_tensor, w_rescale, h_rescale, thresh_iou, thresh_F1, size_group, type_, obj_count):
    TP = 0
    # calculate iou
    giou = grec_box_iou_poly(pred_boxes_tensor, gt_boxes_tensor, w_rescale, h_rescale)
    num_prediction = pred_boxes_tensor.shape[0]
    num_gt = gt_boxes_tensor.shape[0]

    for i in range(min(num_prediction, num_gt)):
        top_value, top_index = torch.topk(giou.flatten(0, 1), 1) # giou between every pred box and gt box
        if top_value < thresh_iou:
            break
        else:
            top_index_x = top_index // num_gt # position before giou flatten
            top_index_y = top_index % num_gt
            TP += 1
            giou[top_index_x[0], :] = 0.0 #filter
            giou[:, top_index_y[0]] = 0.0
    FP = num_prediction - TP
    FN = num_gt - TP
    F_1 = 2 * TP / (2 * TP + FP + FN)

    if F_1 >= thresh_F1:
        # correct_image += 1
        stats['overall']["correct"] += 1
        stats['size'][size_group]['correct'] += 1
        stats['type'][type_]['correct'] += 1
        stats['obj_count'][obj_count]['correct'] += 1
    # num_image += 1
    # updata
    stats['size'][size_group]['total'] += 1
    stats['type'][type_]['total'] += 1
    stats['obj_count'][obj_count]['total'] += 1
    return stats

def compute_geochat_referring_acc(data_pipe_list, prompt_type='json'):
    thresh_iou = 0.5
    thresh_F1 = 1.
    stats = {
        'size': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'obj_count': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'overall': {'correct': 0, 'total': 0}
    }
    for sample in data_pipe_list:
        # get detailed info
        size_group = sample.get("size_group", "unknown")
        type_ = sample.get("type", "unknown")
        obj_count = "single" if len(sample.get("obj_ids", [])) == 1 else "multiple"  
        # updata overall
        stats['overall']['total'] += 1
        # get all output bboxes
        if prompt_type == 'json':
            # do output
            json_str = re.sub(r'^```json|\n```$', '', sample["output"], flags=re.DOTALL).strip()
            output_bbox_dict_list = json.loads(json_str)
            output_bbox_list = [item['bbox_2d'] for item in output_bbox_dict_list]  # TODO: @shuran
            w_rescale = sample["original_width"] / sample["resized_width"]
            h_rescale = sample["original_height"] / sample["resized_height"]
            # do gt
            gt_bbox_list = sample["ground_truth"]
            # flat
            pred_boxes_tensor = adaptive_flatten(torch.tensor(output_bbox_list, dtype=torch.float32))
            gt_boxes_tensor = adaptive_flatten(torch.tensor(gt_bbox_list, dtype=torch.float32))
            stats = get_grec_stats_single_sample(stats, pred_boxes_tensor, gt_boxes_tensor, w_rescale, h_rescale, thresh_iou, thresh_F1, size_group, type_, obj_count)

        elif prompt_type == 'plain':
            pattern4 = r"(\d+),(\d+),(\d+),(\d+)" # zero-shot
            pattern8 = r"(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)" 
            bbox_match = re.findall(pattern8, sample["output"])
            if len(bbox_match) == 0:
                bbox_match = re.findall(pattern4, sample["output"])
            output_bbox_list = [[int(c) for c in coords] for coords in bbox_match]
            w_rescale = sample["original_width"] / sample["resized_width"]
            h_rescale = sample["original_height"] / sample["resized_height"]
            # do gt
            gt_bbox_list = sample["ground_truth"]
            # flat
            pred_boxes_tensor = adaptive_flatten(torch.tensor(output_bbox_list, dtype=torch.float32))  # [box num, coords num]
            gt_boxes_tensor = adaptive_flatten(torch.tensor(gt_bbox_list, dtype=torch.float32))
            stats = get_grec_stats_single_sample(stats, pred_boxes_tensor, gt_boxes_tensor, w_rescale, h_rescale, thresh_iou, thresh_F1, size_group, type_, obj_count)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    results = {
    'overall': {
        "Pr(F1=1, IoU=0.5)": stats['overall']["correct"] / stats['overall']["total"],
        **stats['overall']  # 
    }
    }

    # add 'size'
    for size in stats['size']:
        results[f'size_{size}'] = {
            'Pr(F1=1, IoU=0.5)': stats['size'][size]['correct'] / stats['size'][size]['total'],
            'total_samples': stats['size'][size]['total']
        }
    
    # add 'type'
    for type_ in stats['type']:
        results[f'type_{type_}'] = {
            'Pr(F1=1, IoU=0.5)': stats['type'][type_]['correct'] / stats['type'][type_]['total'],
            'total_samples': stats['type'][type_]['total']
        }
    
    # add 'obj_count'
    for count in stats['obj_count']:
        results[f'count_{count}'] = {
            'Pr(F1=1, IoU=0.5)': stats['obj_count'][count]['correct'] / stats['obj_count'][count]['total'],
            'total_samples': stats['obj_count'][count]['total']
        }
    
    return results

def compute_res_acc(data_pipe_list):
    return 

def geochat_scores(data_pipe_list, clean_benchmark, prompt_type):
    if clean_benchmark in ['geochat_aid', 'geochat_ucmerced']:
        scores = compute_contain_acc(data_pipe_list)
    elif clean_benchmark in ["geochat_region_captioning", "geochat_grounding_description"]:
        scores = compute_cider(data_pipe_list)
    elif clean_benchmark in ["geochat_hrben", "geochat_lrben"]:
        scores = compute_contain_acc(data_pipe_list)
    elif clean_benchmark in ["geochat_referring"]:
        scores = compute_geochat_referring_acc(data_pipe_list, prompt_type)
        # pass # TODO: complete this
    return scores
