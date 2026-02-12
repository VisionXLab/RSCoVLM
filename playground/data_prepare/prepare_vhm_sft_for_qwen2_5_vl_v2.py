import os
import json
from copy import deepcopy

def convert_conversations_to_messages(meta):

    new_meta = deepcopy(meta)
    # rename
    if 'image' in new_meta:
        new_meta['image_id'] = new_meta.pop('image')
    if 'source_dataset' in new_meta:
        new_meta['dataset'] = new_meta.pop('source_dataset')
    messages = []
    image_file_path = new_meta["image_id"]
    # reformat
    messages.append({
        "role": "system",
        "content": "You are an AI assistant."
    })
    
    image_added = False
    
    for turn in new_meta["conversations"]: # support Multi round talk
        if turn["from"] == "human":
            content = []
            if not image_added and "<image>" in turn["value"]:
                content.append({"type": "image", "image": f"file://{image_file_path}"})
                text = turn["value"].replace("<image>", "").strip()
                if text:
                    content.append({"type": "text", "text": text})
                image_added = True
            else:
                content.append({"type": "text", "text": turn["value"]})
            messages.append({"role": "user", "content": content})
            
        elif turn["from"] == "gpt":
            messages.append({"role": "assistant", "content": turn["value"]})

    new_meta["messages"] = messages
    del new_meta["conversations"]  # remove
    
    return new_meta
def convert_conversation(meta):
    # breakpoint()
    meta["image"] = os.path.join(meta["source_dataset"], meta["image"])
    del meta["source_dataset"]
    return meta
if __name__ == "__main__":
    llava_json_list = [
        "playground/data/vhm_sft_mix_qwen2_5_vl_conversations.json",
    ]
    output_json_list = [
        "playground/data/msr_vhm_sft_mix_qwen2_5_vl_conversations.json"
    ]
    idx = 0
    for (json_file, out_json_file) in zip(llava_json_list, output_json_list):
        print(f'Processing {json_file}...')
        # create output directory
        out_root = os.path.dirname(out_json_file)
        os.makedirs(out_root, exist_ok=True)

        data_dict_list = json.load(open(json_file, "r"))
        
        converted_messages = []
        for meta in data_dict_list: 
            # converted_msg = convert_conversations_to_messages(meta) 
            converted_msg = convert_conversation(meta) 
            converted_messages.append(converted_msg)
        
        with open(out_json_file, 'w', encoding='utf-8') as f:  
            json.dump(converted_messages, f, indent=2, ensure_ascii=False)
        
        idx += 1
        print(f"{idx}/{len(llava_json_list)} Finish: saved to {out_json_file}")