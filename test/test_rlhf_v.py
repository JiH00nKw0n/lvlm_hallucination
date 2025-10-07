import json

from datasets import load_dataset

data = load_dataset("HaoyeZhang/RLHF-V-Dataset")

for example in data["train"]:
    print(example)
    print(example["text"])
    print(json.loads(example["text"]))
    break