import json
from fire import Fire

import torch

from mmengine.fileio import load
from mmengine.evaluator import Evaluator

from mmrotate.utils import register_all_modules
from mmdet.utils import register_all_modules as register_all_modules_mmdet


def monkey_patch_of_collections_typehint_for_mmrotate1x():
    import collections
    from collections.abc import Mapping, Sequence, Iterable
    collections.Mapping = Mapping
    collections.Sequence = Sequence
    collections.Iterable = Iterable

monkey_patch_of_collections_typehint_for_mmrotate1x()

register_all_modules_mmdet(init_default_scope=False)
register_all_modules(init_default_scope=False)


def prepare_evaluator(dataset_name, outfile_prefix):
    evaluator_kwargs = dict(
        format_only=True,
        merge_patches=True,
        outfile_prefix=outfile_prefix
    )
    
    if dataset_name == "dota":
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP", **evaluator_kwargs))
        evaluator.dataset_meta = {
            'classes':
            ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
             'harbor', 'swimming-pool', 'helicopter'),
        }
    elif dataset_name == "fair":
        from lmmrotate.modules.fair_metric import FAIRMetric
        evaluator = Evaluator(FAIRMetric(metric="mAP", **evaluator_kwargs))
        evaluator.dataset_meta = {
            'classes':
            ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
            'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
            'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
            'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
            'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
            'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
            'Baseball Field', 'Intersection', 'Roundabout', 'Bridge'),
        }
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    return evaluator


def export_new_submission(pickle_result_path, threshold, outfile_prefix, dataset_name="dota"):
    evaluator = prepare_evaluator(dataset_name, outfile_prefix)
    results_test = load(pickle_result_path)   
    for res in results_test:
        keep = res["pred_instances"]["scores"] > threshold
        res["pred_instances"]["labels"] = res["pred_instances"]["labels"][keep]
        res["pred_instances"]["bboxes"] = res["pred_instances"]["bboxes"][keep]
        res["pred_instances"]["scores"] = torch.ones_like(res["pred_instances"]["scores"][keep])
            
    evaluator.offline_evaluate(data_samples=results_test, chunk_size=128)


if __name__ == "__main__":
    Fire(export_new_submission)