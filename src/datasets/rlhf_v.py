import json
from typing import Optional

from datasets import load_dataset, Dataset, Features, Sequence, Value, Image

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder('RLHFVDatasetBuilder')
class RLHFVDatasetBuilder(BaseBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'rlhf_v'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            path="openbmb/RLHF-V-Dataset",
            split=self.split
        )

        # Define transformation function
        def transform_example(example):
            image = example.get("image")
            data = json.loads(example.get("text"))
            question = data.get("question").strip()
            chosen = data.get("chosen").strip()
            rejected = data.get("rejected").strip()

            return {
                "images": [image],  # Wrap in list
                "prompt": [{"role": "user", "content": question}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}]
            }

        # Apply transformation
        dataset = dataset.map(transform_example, remove_columns=dataset.column_names)

        # Define new features
        new_features = Features(
            {
                "images": Sequence(Image()),
                "prompt": [{
                    "role": Value("string"),
                    "content": Value("string")
                }],
                "chosen": [{
                    "role": Value("string"),
                    "content": Value("string")
                }],
                "rejected": [{
                    "role": Value("string"),
                    "content": Value("string")
                }]
            }
        )

        # Cast to new features
        dataset = dataset.cast(new_features)

        return dataset
