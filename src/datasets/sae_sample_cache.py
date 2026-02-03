from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder("SaeSampleCacheDatasetBuilder")
class SaeSampleCacheDatasetBuilder(BaseBuilder):
    """
    Dataset builder for lmms-lab/sae-sample-cache-dataset.
    """

    split: str = "train"
    streaming: bool = False

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            path="lmms-lab/sae-sample-cache-dataset",
            name=None,
            split=self.split,
            streaming=self.streaming,
        )
        return dataset
