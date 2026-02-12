"""
pip install gradio    # proxy_on first
python /mnt/petrelfs/liqingyun/LLaVA-Geo/scripts_py/vis_app.py
# browse data in http://localhost:10066/
"""
import io
import os
import json
import base64
import gradio as gr
from PIL import Image


llava_data_root = "/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data"
geochat_data_root = "/mnt/petrelfs/share_data/liqingyun/datasets/geochat_data"
fitrs_data_root = "/mnt/petrelfs/share_data/liqingyun/datasets/FIT-RS"
rsgpt_data_root = "/mnt/petrelfs/share_data/liqingyun/datasets/rsgpt_dataset"
skyeyegpt_data_root = "/mnt/petrelfs/share_data/liqingyun/datasets/SkyEye-968k"
geochat_filepath = {
    "geochat_data_root": [
        f"{geochat_data_root}/GeoChat_Instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_nwpu_resisc45_repaired_cls": [
        f"{geochat_data_root}/unmixed_data/nwpu_resisc45_repaired_cls_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_lrben_vqa": [
        f"{geochat_data_root}/unmixed_data/lrben_vqa_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_floodnet_vqa": [
        f"{geochat_data_root}/unmixed_data/floodnet_vqa_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_grounding_tokens": [
        f"{geochat_data_root}/unmixed_data/grounding_tokens_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_identify_tokens": [
        f"{geochat_data_root}/unmixed_data/identify_tokens_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_refer_tokens": [
        f"{geochat_data_root}/unmixed_data/refer_tokens_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_multi_round_conversation": [
        f"{geochat_data_root}/unmixed_data/multi_round_conversation_instruct.json", 
        f"{geochat_data_root}/images",
    ],
    "geochat_others": [
        f"{geochat_data_root}/unmixed_data/others_instruct.json", 
        f"{geochat_data_root}/images",
    ],
}
skysensegpt_filepath = {
    # FIT-RSRC Bench
    "FIT-RSRC_Questions_2k": (
        f"{fitrs_data_root}/FIT-RSRC/Images", 
        f"{fitrs_data_root}/FIT-RSRC/FIT-RSRC_Questions_2k.jsonl"    
    ), 
    # FIT-RSFG Bench
    "FITRSFG_complex_comprehension": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RSFG/FIT-RSFG-Bench/test_FITRS_complex_comprehension_eval.jsonl"
    ),
    "FITRSFG_region_caption": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RSFG/FIT-RSFG-Bench/test_FITRS_region_caption_eval.jsonl"
    ),
    "FITRSFG_image_caption": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RSFG/FIT-RSFG-Bench/test_FITRS_image_caption_eval.jsonl"
    ),
    "FITRSFG_imageclassify": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RSFG/FIT-RSFG-Bench/test_FITRS_imageclassify_eval.jsonl"
    ),
    "FITRSFG_vqa": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RSFG/FIT-RSFG-Bench/test_FITRS_vqa_eval.jsonl"
    ),
    # FIT-RS training
    "FIT-RS-train-1415k": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/FIT-RS-train-1415k.json"    
    ), 
    "FIT-RS-train-sample-381k": (
        f"{fitrs_data_root}/FIT-RS_Image/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/FIT-RS-train-sample-381k.json"    
    ), 
    "imagecaption_65k": (
        f"{fitrs_data_root}/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/train_data_of_each_individual_task/train_instruction_imagecaption_65k.json"
    ), 
    "imageclassification_130k": (
        f"{fitrs_data_root}/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/train_data_of_each_individual_task/train_instruction_imageclassification_130k.json"
    ), 
    "multiturn_50k": (
        f"{fitrs_data_root}/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/train_data_of_each_individual_task/train_instruction_multiturn_50k.json"
    ), 
    "regioncaption_72k": (
        f"{fitrs_data_root}/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/train_data_of_each_individual_task/train_instruction_regioncaption_72k.json"
    ), 
    "vqa_400k": (
        f"{fitrs_data_root}/imgv2_split_512_100_vaild", 
        f"{fitrs_data_root}/FIT-RS_Instruction/train_data_of_each_individual_task/train_instruction_vqa_400k.json"
    ),
}
rsgpt_filepath = {
    "rsicap3k": (
        f"{rsgpt_data_root}/RSICap/images", 
        f"{rsgpt_data_root}/RSICap/llava_rsgpt-rsicap3k.json", 
    ),
    "rsieval-qa": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa.json", 
    ),
    "rsieval-caption": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-caption.json", 
    ),
    "rsieval-qa-ab_position": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-ab_position.json", 
    ),
    "rsieval-qa-area_comp": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-area_comp.json", 
    ),
    "rsieval-qa-color": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-color.json", 
    ),
    "rsieval-qa-image": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-image.json", 
    ),
    "rsieval-qa-presence": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-presence.json", 
    ),
    "rsieval-qa-quantity": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-quantity.json", 
    ),
    "rsieval-qa-reasoning": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-reasoning.json", 
    ),
    "rsieval-qa-re_position": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-re_position.json", 
    ),
    "rsieval-qa-road_ori": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-road_ori.json", 
    ),
    "rsieval-qa-scene": (
        f"{rsgpt_data_root}/RSIEval/images", 
        f"{rsgpt_data_root}/RSIEval/llava_rsgpt-rsieval-qa-scene.json", 
    ),
}
skyeyegpt_filepath = {
    # captions
    "skyeye_cap_ERA": (
        "/mnt/petrelfs/liqingyun/share_data/datasets/EAR/SingleFrames/Tra",
        f"{skyeyegpt_data_root}/caption/caption_ERA_train.json"
    ),
    "skyeye_cap_RSICD": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rs_caption/rsicd/RSICD_images", 
        f"{skyeyegpt_data_root}/caption/caption_RSICD_train.json"    
    ),
    "skyeye_cap_Sydney": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rs_caption/sydney_caption/imgs", 
        f"{skyeyegpt_data_root}/caption/caption_Sydney_train.json"    
    ),
    "skyeye_cap_NWPU": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/NWPU-RESISC45", 
        f"{skyeyegpt_data_root}/caption/caption_NWPU_train.json"
    ),
    "skyeye_cap_RSITMD": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rs_caption/rsitmd/images", 
        f"{skyeyegpt_data_root}/caption/caption_RSITMD_train.json"
    ),
    "skyeye_cap_UCM": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/UCMerced", 
        f"{skyeyegpt_data_root}/caption/caption_UCM_train.json"
    ),
    # multitask_conversation
    "skyeye_mtc_DIOR": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/DIOR/JPEGImages-trainval", 
        f"{skyeyegpt_data_root}/multitask_conversation/DIOR_Conversa.json"
    ),
    "skyeye_mtc_DOTA": (
        "TODO", 
        f"{skyeyegpt_data_root}/multitask_conversation/DOTA_Conversa.json"
    ),
    "skyeye_mtc_Sydney": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rs_caption/sydney_caption/imgs", 
        f"{skyeyegpt_data_root}/multitask_conversation/Sydney_Conversa.json"
    ),
    "skyeye_mtc_UCM": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/UCMerced", 
        f"{skyeyegpt_data_root}/multitask_conversation/UCM_Conversa.json"
    ),
    # phrasegrounding
    "skyeye_phrasegrounding_DIOR": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/DIOR/JPEGImages-trainval", 
        f"{skyeyegpt_data_root}/phrasegrounding/diorpg_train.json"
    ),
    # vg
    "skyeye_rsvg": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/RSVG/images", 
        f"{skyeyegpt_data_root}/vg/rsvg.json"
    ),
    "skyeye_diorrsvg": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/DIOR/JPEGImages-trainval", 
        f"{skyeyegpt_data_root}/vg/DIORvg_train.json"
    ),
    # vqa
    "skyeye_vqa_dota": (
        "TODO", 
        f"{skyeyegpt_data_root}/vqa/DOTAvqa_train.json"
    ),
    "skyeye_vqa_ear": (
        "/mnt/petrelfs/liqingyun/share_data/datasets/EAR/SingleFrames/Tra", 
        f"{skyeyegpt_data_root}/vqa/EARVqa.json"
    ),
    "skyeye_vqa_rsvqahr": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rsvqa/RSVQA_HR/Data", 
        f"{skyeyegpt_data_root}/vqa/RSVQAHR_train.json"
    ),
    "skyeye_vqa_rsvqalr": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rsvqa/RSVQA_LR/Images_LR", 
        f"{skyeyegpt_data_root}/vqa/RSVQALR_train.json"
    ),
    "skyeye_vqa_sydney": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/rs_caption/sydney_caption/imgs", 
        f"{skyeyegpt_data_root}/vqa/SydneyVqa.json"
    ),
    "skyeye_vqa_ucm": (
        "/mnt/petrelfs/share_data/liqingyun/datasets/UCMerced", 
        f"{skyeyegpt_data_root}/vqa/UCMvqa.json"
    ),
}
lhrsbot_path = {
    
}
filepath = {
    **geochat_filepath, 
    **rsgpt_filepath, 
    **skysensegpt_filepath, 
    **skyeyegpt_filepath, 
    **lhrsbot_path, 
    "llava_v1_5_mix665k": [
        f"{llava_data_root}/llava_v1_5_mix/data_root",
        f"{llava_data_root}/llava_v1_5_mix/llava_v1_5_mix665k.json", 
    ],
}


class APPFUNC:
    def __init__(self, filepath_dict):
        self.filepath_dict = filepath_dict
        self.data, self.loaded_obj = None, {}
        self.img_root, self.data_path = None, None
    
    @staticmethod
    def print(*args):
        if True:  # set VERBOSE
            print(*args)
    
    def load_and_collate_annotations(self, ann_filename):
        self.print("Calling load_and_collate_annotations")
        self.img_root, self.data_path = self.filepath_dict[ann_filename]
        print(f"Loading {ann_filename}: {self.data_path} | {self.img_root}")
        if self.data_path.endswith(".json"):
            dataset = json.load(open(self.data_path, "r"))
        else:
            dataset = [json.loads(_) for _ in open(self.data_path, "r").readlines()]
        print("The dataset has been loaded.")
        return dataset
    
    def when_btn_submit_click(self, ann_filename, ann_id, md_annotation):
        self.print("Calling when_btn_submit_click")
        if ann_filename is None:
            return self.when_ann_filename_change(ann_filename, ann_id, md_annotation)
        try:
            item = self.data[int(max(min(ann_id, len(self.data) - 1), 0))]
        except IndexError as err:
            print(ann_id, len(self.data), int(max(min(ann_id, len(self.data) - 1), 0)))
            raise err
        md_annotation = self.item2md(item)
        return ann_filename, int(max(min(ann_id, len(self.data) - 1), 0)), md_annotation
    
    def when_btn_next_click(self, ann_filename, ann_id, md_annotation):
        self.print("Calling when_btn_next_click")
        return self.when_btn_submit_click(ann_filename, ann_id + 1, md_annotation)
    
    def when_ann_filename_change(self, ann_filename, ann_id, annotation):
        self.print("Calling when_ann_filename_change")
        if ann_filename not in self.filepath_dict:
            return ann_filename, ann_id, annotation
        obj = self.loaded_obj.get(ann_filename, None) 
        if obj is None:
            obj = self.loaded_obj[ann_filename] = self.load_and_collate_annotations(ann_filename)
        self.data = obj
        return self.when_btn_submit_click(ann_filename, 0, annotation)
    
    def item2md(self, item):
        image = item.get("image", None)
        text = item.get("text", None)
        ground_truth = item.get("ground_truth", None)
        conversations = item.get("conversations", None)
        
        md_str = ""
        
        if image is not None:
            assert isinstance(image, str)
            img_path = os.path.join(self.img_root, image) if self.img_root is not None else image
            pil_img = Image.open(img_path)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            base64_img = base64.b64encode(img_byte_arr).decode("utf-8")
            img_md = f"![{image}](data:image/png;base64,{base64_img})"
            md_str += f"## Image:\n{img_md}\n\n"
        
        if text is not None:
            assert isinstance(text, str)
            md_str += f"## Text:\n{text}\n\n"
            
        if ground_truth is not None:
            assert isinstance(ground_truth, str)
            md_str += f"## Ground Truth:\n{ground_truth}\n\n"
            
        if conversations is not None:
            assert isinstance(conversations, list)
            md_str += f"## Conversations:\n"
            for _c in conversations:
                assert isinstance(_c, dict) and "from" in _c and "value" in _c
                md_str += f"- **`{_c['from']}`**: {_c['value']}\n"
            md_str += "\n\n"

        md_str += "\n".join([
            "\n\n## META DATA DICT: ",
            "```", 
            f"meta_data = {json.dumps(item, indent=4)}", "", "", 
            "```",  
        ])
        return md_str


def gradio_vis_app(_filepath):
    app_func = APPFUNC(_filepath)
        
    with gr.Blocks() as app:
        ann_filename = gr.Radio(list(_filepath.keys()), value=None)
        with gr.Row():
            ann_id = gr.Number(0)
            btn_next = gr.Button("Next")
            btn_submit = gr.Button("id跳转")
        annotation = gr.Markdown()
        
        all_components = [ann_filename, ann_id, annotation]
        ann_filename.change(app_func.when_ann_filename_change, all_components, all_components)
        btn_submit.click(app_func.when_btn_submit_click, all_components, all_components)
        btn_next.click(app_func.when_btn_next_click, all_components, all_components)
        
    # app.launch()
    app.launch(server_name="0.0.0.0", share=True, server_port=10098)
        

if __name__ == "__main__":
    gradio_vis_app(filepath)