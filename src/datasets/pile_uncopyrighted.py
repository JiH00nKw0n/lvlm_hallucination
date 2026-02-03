from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder("PileUncopyrightedDatasetBuilder")
class PileUncopyrightedDatasetBuilder(BaseBuilder):
    """
    Dataset builder for monology/pile-uncopyrighted.
    """

    split: str = "train"
    streaming: bool = True

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            path="monology/pile-uncopyrighted",
            name=None,
            split=self.split,
            streaming=self.streaming,
        )
        return dataset
