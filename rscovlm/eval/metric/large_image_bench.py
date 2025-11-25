"""
python -u -m rscovlm.eval.metric.large_image_bench --result_path /path/to/result.jsonl --result_compare_path /path/to/compare_result.jsonl --acc_method contain
"""

from .metric import compute_contain_acc, compute_startswith_acc


def compute_wups_acc(data_pipe_list):
    from nltk.corpus import wordnet as wn

    def are_synonyms(word1, word2):
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        for synset1 in synsets1:
            for synset2 in synsets2:
                if synset1.path_similarity(synset2) is not None and synset1.path_similarity(synset2) > 0.8:
                    return True
        return False

    correct = 0
    
    for item in data_pipe_list:
        gt = item["ground_truth"].lower().strip().strip(".")
        pred = item["output"].lower().strip().strip(".")
        
        if gt == pred:
            correct += 1
        elif are_synonyms(gt, pred):
            correct += 1
    
    if len(data_pipe_list) == 0:
        return None
    else:
        acc = correct / len(data_pipe_list)
        return {"acc": acc}


def mme_realworld_remote_sensing_scores(data_pipe_list):
    return {
        "acc": compute_contain_acc(data_pipe_list)["acc"],
        "count_acc": compute_contain_acc([dp for dp in data_pipe_list if dp['Category'] == 'count'])["acc"],
        "color_acc": compute_contain_acc([dp for dp in data_pipe_list if dp['Category'] == 'color'])["acc"],
        "position_acc": compute_contain_acc([dp for dp in data_pipe_list if dp['Category'] == 'position'])["acc"],
    }


def lrsvqa_scores(data_pipe_list):
    results = {"overall acc": compute_wups_acc(data_pipe_list)["acc"]}

    sources = ('FAIR', 'STAR', 'GLH')
    tasks = ('rural or urban', 'object shape', 'object status', 'count', 'object background', 'object color', 'object category', 'reasoning')

    filter_source = lambda ls, src: [dp for dp in ls if dp['question_id'].startswith(src)]
    filter_task = lambda ls, task: [dp for dp in ls if dp['category'] == task]
    sources_oa = {f"Overall Acc (Source {src})": compute_wups_acc(filter_source(data_pipe_list, src))['acc'] for src in sources}
    tasks_oa = {f"Overall Acc (Task {task})": compute_wups_acc(filter_task(data_pipe_list, task))['acc'] for task in tasks}
    results.update({**sources_oa, **tasks_oa})

    filter_source_task = lambda ls, src, task: [dp for dp in ls if dp['question_id'].startswith(src) and dp['category'] == task]
    for src in sources:
        r = [compute_wups_acc(filter_source_task(data_pipe_list, src, task)) for task in tasks]
        r = [_["acc"] for _ in r if _ is not None]
        results[f"Average Acc (Source {src})"] = sum(r) / len(r)
    r = [compute_wups_acc(filter_task(data_pipe_list, task)) for task in tasks]
    r = [_["acc"] for _ in r if _ is not None]
    results[f"Average Acc"] = sum(r) / len(r)
    return results


def visualize_results(result_path, result_compare_path=None, acc_method='contain', min_pixels=None, max_pixels=None):
    import io
    import json
    import base64
    import gradio as gr
    from PIL import ImageDraw
    from qwen_vl_utils.vision_process import fetch_image

    title_msg = "# Large Image Bench Results Visualization\n\n"
    title_msg += f" (Acc. Method: {acc_method})\n\n"
    title_msg += f" (Result Path: {result_path})\n\n"
    if result_compare_path is not None:
        title_msg += f" (Compare Path: {result_compare_path})\n\n"

    img_ele_kwargs = {}
    if min_pixels is not None:
        img_ele_kwargs['min_pixels'] = min_pixels
    if max_pixels is not None:
        img_ele_kwargs['max_pixels'] = max_pixels

    compute_acc = compute_contain_acc if acc_method == 'contain' else compute_wups_acc
    results = json.load(open(result_path, 'r'))
    if result_compare_path is not None:
        result_compare = {}
        for sample in json.load(open(result_compare_path, 'r')):
            result_compare[sample['idx']] = sample
        
        sort_key_num = {}
        for result in results:
            result['compare'] = result_compare[result['idx']]

            result['acc'] = compute_acc([result])['acc']
            result['compare_acc'] = compute_acc([result['compare']])['acc']
            result['sort_key'] = (
                len(result['message']) + len(result['compare']['message']), 
                2 * result['compare_acc'] - result['acc'] - 1, 
                -result['idx']
            )

            if result['sort_key'][:2] not in sort_key_num:
                sort_key_num[result['sort_key'][:2]] = 0
            sort_key_num[result['sort_key'][:2]] += 1

        title_msg += f"Diff Num: {len(sort_key_num)}\n\n"
        start_idx = 0
        for sort_key in sorted(sort_key_num.keys(), reverse=True):
            end_idx = start_idx + sort_key_num[sort_key]
            title_msg += f"\t{sort_key}: {sort_key_num[sort_key]} {start_idx} --> {end_idx}\n\n"
            start_idx = end_idx
    else:
        for result in results:
            result['acc'] = metric([result])['acc']
            result['sort_key'] = (len(result['message']), result['acc'], -result['idx'])
            
    results.sort(key=lambda x: x['sort_key'], reverse=True)

    def get_img_md(pil_image, thumbnail_size=(512, 512)):
        buffered = io.BytesIO()
        h, w = pil_image.size
        thumbnail = pil_image.copy()
        thumbnail.thumbnail(thumbnail_size)
        thumbnail.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"![Image](data:image/png;base64,{img_str})\n\n"

    def when_btn_jump_clicked(idx, res):
        idx = max(0, min(len(results) - 1, int(idx)))
        result = results[idx]

        res = ""
        res += f"## Result {idx}:\n\n"
        message = result['message']
        if len(result['compare']['message']) > len(message):
            message = result['compare']['message']

        for i, msg in enumerate(message):
            role = msg['role']
            content = msg['content']
            res += f"### {role}\n\n"

            if isinstance(content, str):
                if '<tool_call>' in content:
                    res += f"```\n{content}\n```\n\n"
                else:
                    res += content.replace('\n', '\n\n') + "\n\n"
            elif isinstance(content, list):
                for item in content:
                    if item['type'] == 'text':
                        res += item['text'].replace('\n', '\n\n') + "\n\n"
                    else:
                        if item['type'] == 'image' and item.get('image', '').startswith('<PIL.Image.Image'):
                            tool_call = json.loads(
                                message[i - 1]['content']
                                .lstrip('</tool_call>\n')
                                .rstrip('\n</tool_call>')
                            )
                            assert tool_call['name'] == 'image_zoom_in', tool_call
                            bbox = tool_call['arguments']['bbox']

                            whole_image = pil_image.copy()
                            ImageDraw.Draw(whole_image).rectangle(bbox, outline="red", width=5)
                            res += get_img_md(whole_image)

                            cropped_image = pil_image.crop(bbox)
                            res += get_img_md(cropped_image)
                        else:
                            pil_image = fetch_image({**item, **img_ele_kwargs})
                            res += get_img_md(pil_image)
            else:
                raise ValueError(f"Unexpected content: {content}")

        res += f"## ground_truth:\n\n {result['ground_truth']}\n\n"
        res += f"## output:\n\n {result['output']}\n\n"
        res += f"## output (compare):\n\n {result['compare']['output']}\n\n"

        res += f"## Metadata:\n\n"
        res += f"```json\n{json.dumps(result, indent=4)}\n```\n\n"
        return idx, res

    def when_btn_prev_clicked(idx, res):
        idx = max(0, int(idx) - 1)
        return when_btn_jump_clicked(idx, res)

    def when_btn_next_clicked(idx, res):
        idx = min(len(results) - 1, int(idx) + 1)
        return when_btn_jump_clicked(idx, res)

    with gr.Blocks() as app:
        gr.Markdown(title_msg)
        idx = gr.Number(label="Index", value=0, precision=0, step=1, minimum=0, maximum=len(results)-1)
        with gr.Row():
            btn_jump = gr.Button("Jump to Index")
            btn_prev = gr.Button("Previous")
            btn_next = gr.Button("Next")

        res = gr.Markdown(label="Result", value="")

        btn_jump.click(when_btn_jump_clicked, inputs=[idx, res], outputs=[idx, res])
        btn_prev.click(when_btn_prev_clicked, inputs=[idx, res], outputs=[idx, res])
        btn_next.click(when_btn_next_clicked, inputs=[idx, res], outputs=[idx, res])

    app.launch(share=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Large Image Bench Results")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result file")
    parser.add_argument("--result_compare_path", type=str, default=None, help="Path to the comparison result file")
    parser.add_argument("--acc_method", type=str, default='contain', choices=['contain', 'wups', 'startswith'], help="Method to compute accuracy")
    parser.add_argument("--min_pixels", type=int, default=448 ** 2, help="Minimum number of pixels for filtering results")
    parser.add_argument("--max_pixels", type=int, default=1008 ** 2, help="Maximum number of pixels for filtering results")
    args = parser.parse_args()

    visualize_results(args.result_path, args.result_compare_path, args.acc_method, args.min_pixels, args.max_pixels)
