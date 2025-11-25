import random
from . import prepare_data_pipe
from ..params import parse_args
from qwen_vl_utils import process_vision_info


def test_check_loading_data():
    benchmarks_to_check = (
        # zero-shot classification
        "cls_aid",
        "cls_resisc_timm",
        "cls_eurosat_timm",

        # zero-shot VRSBench
        "vrsbench_caption",
        "vrsbench_vqa",

        # zero-shot image caption
        "cap_nwpu_caption",
        "cap_rsicd",
        "cap_rsitmd",
        "cap_sydney_caption",
        "cap_ucm_caption",

        # # few-shot image caption
        # "cap_nwpu_caption_16",
        # "cap_rsitmd_16",
        # "cap_sydney_caption_16",
        # "cap_ucm_caption_16",

        # remote sensing grounding
        "dior_rsvg_val",

        # # general grounding
        # "refcoco_val",
        # "refcocop_val",
        # "refcocog_val",

        # large remote sensing images benchmark
        "mme_realworld_remote_sensing",
        "lrsvqa",

        # zero-shot VHM Bench
        'vhm_open_ended_qa_full', 
        'vhm_hnstd_qa'
    )
    args = parse_args()

    errors = {}
    for benchmark in benchmarks_to_check:
        try:
            list_data_pipe = random.sample(prepare_data_pipe(benchmark, args), 3)

            for data_pipe in list_data_pipe:
                message = data_pipe['message']
                image_inputs, _ = process_vision_info(message)
                print(f"loaded image_inputs sizes: {[img.size for img in image_inputs]}")

            print(f"Successfully loading {benchmark} data")
        except Exception as err:
            errors[benchmark] = str(err)
            print(f"Error loading {benchmark}: {err}")

    if len(errors) > 0:
        print("Errors occurred while loading the following benchmarks:")
        for benchmark, err in errors.items():
            print(f"{benchmark}: {err}")
        with open("tmp_eval_data_error.log", "w") as f:
            for benchmark, err in errors.items():
                f.write(f"{benchmark}: {err}\n")


if __name__ == '__main__':
    test_check_loading_data()
