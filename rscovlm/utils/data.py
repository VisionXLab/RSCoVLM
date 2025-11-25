import json
import random


def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data


def load_json_or_jsonl(filepath):
    if filepath.endswith('.jsonl'):
        return load_jsonl(filepath)
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as json_file:
            return json.load(json_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")


def random_sample(data, annotations, seed=None):
    if seed is not None:
        seed = 42
    rnd = random.Random(seed)
    
    sampling_rate = data.get("sampling_rate", 1.0)
    if sampling_rate < 1.0:
        annotations = rnd.sample(
            annotations, int(len(annotations) * sampling_rate)
        )
    else:
        assert sampling_rate % 1 == 0, "Sampling rate must be an integer when >= 1.0"
        sampling_rate = int(sampling_rate)
        annotations = annotations * sampling_rate
    return annotations
