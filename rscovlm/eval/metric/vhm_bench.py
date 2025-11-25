import re
from collections import defaultdict


def eval_contain(pred, gt):
    pred = pred.lower().strip().strip(".")
    gt = gt.lower().strip().strip(".")
    return pred in gt or gt in pred


def eval_presence(pred, gt):
    gt = gt.lower().strip().strip(".")
    pred = pred.lower().strip().strip(".")

    x_yes = 'yes' in pred or 'true' in pred
    x_no = 'no' in pred or 'false' in pred

    if x_yes and not x_no:
        pred = "yes"
    elif x_no and not x_yes:
        pred = "no"
    else:
        pred = "unknown"
    return pred == gt


def eval_color(pred, gt):
    pred = pred.lower().strip().strip(".").replace(" ", "")
    gt = gt.lower().strip().strip(".").replace(" ", "").split(",")
    return sum(e in pred for e in gt) / len(gt)


def eval_multi_choices(pred, gt):
    pred = pred.strip().strip(".")
    gt = gt.lower().strip().strip(".")
    pred = next((char.lower() for char in pred if char in 'ABCDE'), None)
    if pred is None:
        return False
    return pred in gt.lower()


def convert_to_words_100(num):
    def one_to_nineteen(n):
        words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
                "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", 
                "Eighteen", "Nineteen"]
        return words[n-1].lower()

    def tens(n):
        words = ["Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        return words[n-2].lower()

    if num == 0:
        return "zero"
    if num < 20:
        return one_to_nineteen(num)
    if num < 100:
        if num % 10 == 0:
            return tens(num // 10)
        else:
            return tens(num // 10) + "-" + one_to_nineteen(num % 10)


def word_to_num_100(sentence):
    # Mapping of number words to their numeral equivalents
    number_words = {convert_to_words_100(i): i for i in range(100)}

    # Split the sentence into words and look for number words
    for word in sentence.split():
        if word.lower() in number_words:
            return str(number_words[word.lower()])
    return None


def parse_number(x):
    if re.search(r'\d+', x):
        return re.search(r'\d+', x).group()
    else:
        extract_number = word_to_num_100(x)
        return extract_number if extract_number is not None else ''


def eval_counting_mae(pred, gt):
    pred = int(parse_number(pred.lower().strip().strip(".")))

    sample_mae = abs(pred - gt)
    sample_mape = abs((pred - gt) / gt)
    return sample_mae, sample_mape


def eval_with_mae(pred, gt, ori_implementation=False):
    EXTRACT_NUMBER_PATTERN = r"[-+]?\d*\.?\d+"
    gt = re.findall(EXTRACT_NUMBER_PATTERN, gt)
    pred = re.findall(EXTRACT_NUMBER_PATTERN, pred)

    if len(gt) != len(pred):
        if not ori_implementation:
            for i, item in enumerate(gt):
                if i < len(pred):
                    yield abs(pred - gt), abs((pred - gt) / gt)
                # else:  # wrong
                #     yield 0, 0
    else:
        for g, p in zip(gt, pred):
            p, g = float(p), float(g)
            yield abs(p - g), abs((p - g) / g)


def extract_cate(text):
    start_index = text.find(':')
    cate_names = [name.strip() for name in text[start_index+1:-1].split(',')]
    return cate_names


def match_cate(text, cate_names):
    match_index = []
    for cate_name in cate_names:
        if cate_name in text:
            match_index.append(1)
        else:
            match_index.append(0)
    return match_index


def compute_multi_label_cls_acp_score(pred_and_gt_list):
    # The paper report mF1, but the official code offer average CP only
    raise NotImplementedError("The function compute_multi_label_cls_acp_score is not implemented yet.")


def compute_vhm_scores(data_pipe_list):
    scores = defaultdict(list)

    for item in data_pipe_list:
        gt = item["ground_truth"]
        pred = item["output"]
        folder_name = item["image_path"]

        if folder_name == "presence_mo_dota":
            scores['presence_acc'].append(eval_presence(pred, gt))

        elif folder_name == "color_dota-test_fair1m-val_open":
            scores['color_acc'].append(eval_color(pred, gt))

        elif folder_name == "abspos_dota-test_mc":
            scores['abspos_position_acc'].append(eval_multi_choices(pred, gt))

        elif folder_name == "relpos_dota-test_mc":
            scores['relpos_position_acc'].append(eval_multi_choices(pred, gt))

        elif folder_name == "counting_dota-test_open":
            sample_mae, sample_mape = eval_counting_mae(pred, gt)
            scores['counting_mae'].append(sample_mae)
            scores['counting_mape'].append(sample_mape)

        elif folder_name == "imgType_mcq":
            scores['modality_acc'].append(eval_multi_choices(pred, gt))

        elif folder_name == "gsd_dota_fbp":
            for sample_mae, sample_mape in eval_with_mae(pred, gt):
                scores['resolution_mae'].append(sample_mae)
                scores['resolution_mape'].append(sample_mape)

        elif folder_name == "obj_meas_dota_test":
            for sample_mae, sample_mape in eval_with_mae(pred, gt):
                scores['geometric_mae'].append(sample_mae)
                scores['geometric_mape'].append(sample_mape)

        elif folder_name.startswith('mlc_'):
            scores['multi_label_cls'].append((pred, gt))

        elif folder_name.startswith('cls') or folder_name.lower().startswith('rsvqa'):
            # if eval_contain(pred, gt) is False:
            #     with open("tmp.txt", "a") as f:
            #         f.write(f"{folder_name}: {pred} | {gt}\n")
            scores[folder_name].append(eval_contain(pred, gt))

        else:
            raise ValueError(f"Unknown folder name: {folder_name}")

    for name, items in scores.items():
        if name == 'multi_label_cls':
            scores[name] = compute_multi_label_cls_acp_score(items)
        else:
            scores[name] = sum(items) / max(len(items), 1)
    return scores
