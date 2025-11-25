"""
The code is modified from https://github.com/VisionXLab/mllm-mmrotate/blob/master/lmmrotate/dataset.py
"""

import os
import json
import random

import torch
from torch.utils.data import Dataset, Subset
from mmdet.structures.bbox import BaseBoxes
from rscovlm.utils import monkey_patch_of_collections_typehint_for_mmrotate1x

monkey_patch_of_collections_typehint_for_mmrotate1x()


class OrientedDetEvalDataset:

    default_data_root = {
        "dota": "./playground/data/detection/split_ss_dota", 
        "dota_512": "./playground/data/detection/split_ss_dota_512", 
        "dior": "./playground/data/detection/DIOR",
        "fair1m": "./playground/data/detection/split_ss_fair1m_1_0", 
        "fair1m_2.0_train": "./playground/data/detection/split_ss_fair1m_2_0",
        "srsdd": "./playground/data/detection/SRSDD",
        "dota_train": "./playground/data/detection/split_ss_dota",
        "rsar": "./playground/data/detection/RSAR",
    }
    
    func_map = {  # dataset_type: (is_test_set, not is_test_set)
        "dota": ("initialize_dota_dataset", "initialize_coco_format_dota_dataset"),
        "dota_512": ("initialize_dota_dataset", "initialize_coco_format_dota_dataset"),
        "dior": ("initialize_coco_format_dior_dataset", "initialize_coco_format_dior_dataset"),
        "fair1m": ("initialize_fair1m_dataset", "initialize_coco_format_fair1m_dataset"),
        "fair1m_2.0_train": ("initialize_coco_format_fair1m_dataset", "initialize_coco_format_fair1m_dataset"),
        "srsdd": ("initialize_coco_format_srsdd_dataset", "initialize_coco_format_srsdd_dataset"),
        "dota_train": ("initialize_dota_dataset", "initialize_dota_dataset"),
        "rsar": ("initialize_rsar_dataset", "initialize_coco_format_rsar_dataset"),
    }

    def __init__(self, dataset_type="dota", data_root=None, shuffle_seed=None, clip_num=None, is_test_set=False, box_type='rbox'):
        self.dataset_type = dataset_type
        self._data_root = data_root
        self.is_test_set = is_test_set
        self.box_type = box_type
        
        from mmdet.utils import register_all_modules as register_mmdet_modules
        from mmrotate.utils import register_all_modules as register_mmrotate_modules
        register_mmdet_modules(init_default_scope=False)
        register_mmrotate_modules(init_default_scope=True)

        self.initialize_dataset()
        
        if clip_num is not None or shuffle_seed is not None:
            indices = list(range(len(self.dataset)))
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(indices)
            if clip_num is not None:
                indices = indices[:clip_num]
            self.dataset = Subset(self.dataset, indices)
        
        self.cls_map = {c.replace("-", " ").lower(): i
            for i, c in enumerate(self.metainfo['classes'])
        }  # in mmdet v2.0 label starts from 0

    @property
    def data_root(self):
        return self._data_root or self.default_data_root[self.dataset_type]

    def initialize_dataset(self):
        func_name = self.func_map[self.dataset_type][int(not self.is_test_set)]
        getattr(self, func_name)()

    def initialize_dota_dataset(self):
        from mmrotate.datasets import DOTADataset
        online_pipeline = [dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))]

        offline_pipeline = [dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox')]
        if self.box_type == 'rbox':
            offline_pipeline.append(dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')))
        offline_pipeline.append(dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name')))

        if self.dataset_type == "dota_train":
            img_prefix = 'trainval/images/'
            ann_file = 'val/annfiles/' if self.is_test_set else 'train/annfiles/'
            pipeline = offline_pipeline
        elif self.dataset_type in ["dota", "dota_512"]:
            img_prefix = 'test/images/' if self.is_test_set else 'trainval/images/'
            ann_file = '' if self.is_test_set else 'trainval/annfiles/'
            pipeline = online_pipeline if self.is_test_set else offline_pipeline
        else:
            raise ValueError(f"dataset_type={self.dataset_type} is not supported.")
        
        self.dataset = DOTADataset(
            data_root=self.data_root, 
            ann_file=ann_file,
            data_prefix=dict(img_path=img_prefix),
            test_mode=True,
            pipeline=pipeline,
        )
        
    def initialize_coco_format_dota_dataset(self):
        "Faster initialized"
        assert not self.is_test_set, "COCO format is not implemented for test set here."
        classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                 'basketball-court', 'storage-tank', 'soccer-ball-field',
                 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file='trainval.json', img_prefix='trainval/images/')

    def initialize_dior_dataset(self):
        from mmrotate.datasets import DIORDataset

        offline_pipeline = [dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox')]
        if self.box_type == 'rbox':
            offline_pipeline.append(dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')))
        offline_pipeline.append(dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name')))

        self.dataset = DIORDataset(
            data_root=self.data_root, 
            ann_file='ImageSets/Main/test.txt' if self.is_test_set else 'ImageSets/Main/trainval.txt',  # you may require `cat train.txt val.txt > trainval.txt` to generate this file
            data_prefix=dict(img_path='JPEGImages-test') if self.is_test_set else dict(img_path='JPEGImages'),
            test_mode=True,
            pipeline=offline_pipeline
        )

    def initialize_coco_format_dior_dataset(self):
        classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
                   'chimney', 'expressway-service-area', 'expressway-toll-station',
                   'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
                   'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
        ann_file = 'Annotations/test.json' if self.is_test_set else 'Annotations/trainval.json'
        img_prefix = 'JPEGImages-test/' if self.is_test_set else 'JPEGImages-trainval/'
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file, img_prefix)

    def initialize_fair1m_dataset(self):
        from lmmrotate.modules.rsar_dataset import RSARDataset
        online_pipeline = [dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))]

        offline_pipeline = [dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox')]
        if self.box_type == 'rbox':
            offline_pipeline.append(dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')))
        offline_pipeline.append(dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name')))

        from lmmrotate.modules.fair_dataset import FAIRDataset
        self.dataset = FAIRDataset(
            data_root=self.data_root, 
            ann_file='' if self.is_test_set else 'train/annfiles/',
            data_prefix=dict(img_path='test/images/') if self.is_test_set else dict(img_path='train/images/'),
            test_mode=True,
            pipeline=online_pipeline if self.is_test_set else offline_pipeline
        )

    def initialize_coco_format_fair1m_dataset(self):
        if self.is_test_set:
            if self.dataset_type == "fair1m_2.0_train":
                ann_file = 'validation/val.json'
                img_prefix = 'validation/images/'
            else:
                assert not self.is_test_set, "COCO format is not implemented for test set here."
        else:
            ann_file = 'train/train.json'
            img_prefix = 'train/images/'
        classes = ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
                   'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
                   'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
                   'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
                   'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
                   'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
                   'Baseball Field', 'Intersection', 'Roundabout', 'Bridge')
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file=ann_file, img_prefix=img_prefix)
    
    def initialize_coco_format_srsdd_dataset(self):
        classes = ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil')
        ann_file = 'test.json' if self.is_test_set else 'train.json'
        img_prefix = 'test/images/' if self.is_test_set else 'train/images/'
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file, img_prefix)

    def initialize_rsar_dataset(self):
        from lmmrotate.modules.rsar_dataset import RSARDataset
        offline_pipeline = [dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox')]
        if self.box_type == 'rbox':
            offline_pipeline.append(dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')))
        offline_pipeline.append(dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name')))

        self.dataset = RSARDataset(
            data_root=self.data_root, 
            ann_file='test/annfiles/' if self.is_test_set else 'trainval/annfiles/',  # you may require `cat train.txt val.txt > trainval.txt` to generate this file
            data_prefix=dict(img_path='test/images/') if self.is_test_set else dict(img_path='trainval/images/'),
            test_mode=True,
            pipeline=offline_pipeline
        )

    def initialize_coco_format_rsar_dataset(self):
        classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
        ann_file = 'test.json' if self.is_test_set else 'trainval.json'
        img_prefix = 'test/images/' if self.is_test_set else 'trainval/images/'
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file, img_prefix)

    def initialize_coco_format_dataset(self, data_root, classes, ann_file, img_prefix):
        from mmdet.datasets import CocoDataset
        from mmdet.datasets.transforms import LoadAnnotations

        class CustomLoadAnnotations(LoadAnnotations):
            def transform(self, results):
                results["ori_shape"] = results['height'], results['width']
                results["file_name"] = os.path.basename(results["img_path"])
                results = super().transform(results)
                return results

        return CocoDataset(
            data_root=data_root, 
            metainfo=dict(classes=classes),
            ann_file=ann_file,
            data_prefix=dict(img=img_prefix),
            test_mode=True,
            pipeline=[
                CustomLoadAnnotations(with_bbox=True, with_mask=True, poly2mask=False),
                dict(type='ConvertMask2BoxType', box_type=self.box_type),
                dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
            ]
        )
    
    @property
    def metainfo(self):
        if hasattr(self.dataset, "metainfo"):
            return self.dataset.metainfo
        else:
            return self.dataset.dataset.metainfo

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]["data_samples"]
        img_path = data_sample.img_path
        return img_path, data_sample
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def samplelist_boxtype2list(batch_data_samples):
    for data_samples in batch_data_samples:
        if "gt_instances" in data_samples:
            bboxes = data_samples.gt_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor.tolist()
            elif isinstance(bboxes, torch.Tensor):
                data_samples.gt_instances.bboxes = bboxes.tolist()
            if hasattr(data_samples.gt_instances, 'labels'):
                data_samples.gt_instances.labels = data_samples.gt_instances.labels.tolist()
        if "pred_instances" in data_samples:
            bboxes = data_samples.pred_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor.tolist()
            elif isinstance(bboxes, torch.Tensor):
                data_samples.pred_instances.bboxes = bboxes.tolist()
            if hasattr(data_samples.pred_instances, 'labels'):
                data_samples.pred_instances.labels = data_samples.pred_instances.labels.tolist()
        if "ignored_instances" in data_samples:
            bboxes = data_samples.ignored_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor.tolist()
            elif isinstance(bboxes, torch.Tensor):
                data_samples.ignored_instances.bboxes = bboxes.tolist()
            if hasattr(data_samples.ignored_instances, 'labels'):
                data_samples.ignored_instances.labels = data_samples.ignored_instances.labels.tolist()


def build_eval_dataset(benchmark, box_type='rbox'):
    if benchmark == "dota_trainval512":
        dataset = OrientedDetEvalDataset('dota_512', is_test_set=False, box_type=box_type)
    elif benchmark == "dota_trainval1024":
        dataset = OrientedDetEvalDataset('dota', is_test_set=False, box_type=box_type)
    elif benchmark == "dota_test512":
        dataset = OrientedDetEvalDataset('dota_512', is_test_set=True, box_type=box_type)
    elif benchmark == "dota_test1024":
        dataset = OrientedDetEvalDataset('dota', is_test_set=True, box_type=box_type)
    else:
        raise ValueError(f"benchmark={benchmark} is not supported.")
    return dataset


def prepare_dense_det_messages(image_file_path, categories_str, prompt_type="json", pred_box_type='qbox'):
    message = None
    if pred_box_type == 'qbox':
        if prompt_type == 'json':
            message = [
                {"role": "system", "content": "You are an AI assistant specializing in oriented object detection. Represent objects with quadrant bbox: the corrdinates of each vertices of the oriented bounding boxes in clock-wise order."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_file_path}"},
                    {"type": "text", "text": f"Locate every item from the category list in the image and output the oriented bbox coordinates in JSON format. The category set includes {categories_str}."}
                ]},
            ]
        elif prompt_type == 'plain':
            message = [
                {"role": "system", "content": "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates of the polygon vertices in plain text format 'x1,y1,x2,y2,x3,y3,x4,y4 object'."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_file_path}"},
                    {"type": "text", "text": f"find the {categories_str}."}
                ]},
            ]
    elif pred_box_type == 'rbox':
        if prompt_type == 'json':
            message = [
                {"role": "system", "content": "You are an AI assistant specializing in oriented object detection. Represent objects with: center (x,y), width (w), height (h), and rotation angle."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_file_path}"},
                    {"type": "text", "text": f"Locate every item from the category list in the image and output the oriented bbox coordinates in JSON format. The category set includes {categories_str}."}
                ]},
            ]
        elif prompt_type == 'plain':
            message = [
                {"role": "system", "content": "As an AI assistant, you specialize in accurate oriented object detection, delivering coordinates in plain text format 'x,y,w,h,a object'."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_file_path}"},
                    {"type": "text", "text": f"find the {categories_str}."}
                ]},
            ]
    elif pred_box_type == 'hbox':
        if prompt_type == 'json':
            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_file_path}"},
                    {"type": "text", "text": f"Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes {categories_str}."}
                ]},
            ]
        elif prompt_type == 'plain':
            message = [
                {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_file_path}"},
                    {"type": "text", "text": f"find the {categories_str}."}
                ]},
            ]
    return message


def prepare_dense_det(benchmark="dota_trainval512", pred_box_type='qbox', eval_box_type='rbox', prompt_type="json"):
    dataset = build_eval_dataset(benchmark, box_type=eval_box_type)

    categories = dataset.metainfo['classes']
    categoryid2rank = {}
    categoryid2name = {}
    cat_names = sorted([_ for _ in categories], key=lambda s: s.replace("-", " ").lower())
    for category_id, category_name in enumerate(categories):
        categoryid2rank[category_id] = cat_names.index(category_name)
        categoryid2name[category_id] = category_name.replace("-", " ").lower()
    categories_str = ', '.join([c.replace("-", " ").lower() for c in cat_names])

    data_pipe_list = []
    for image_file_path, data_sample in dataset:
        instance_num = len(data_sample.gt_instances)
        
        message = prepare_dense_det_messages(
            image_file_path=image_file_path,
            categories_str=categories_str,
            prompt_type=prompt_type, 
            pred_box_type=pred_box_type
        )

        samplelist_boxtype2list([data_sample])
        data_pipe_list.append({
            'message': message, 
            'instance_num': instance_num,
            **data_sample.to_dict()
        })

    if not 'test' in benchmark:
        data_pipe_list.sort(key=lambda x: x['instance_num'])
    return data_pipe_list


if __name__ == '__main__':
    ds = prepare_dense_det('dota_trainval512')
    import ipdb; ipdb.set_trace()
