from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder('RLHFVDatasetBuilder')
class RLHFVDatasetBuilder(BaseBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'rlhf_v'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "openbmb/RLHF-V-Dataset", trust_remote_code=True, split=self.split
        )
        dataset = dataset.rename_columns({"image": 'images'})

        return dataset
