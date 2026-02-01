from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder("LlavaNextDataDatasetBuilder")
class LlavaNextDataDatasetBuilder(BaseBuilder):
    path: str = "lmms-lab/LLaVA-NeXT-Data"
    split: Optional[str] = "train"
    name: Optional[str] = None
    trust_remote_code: bool = False

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            path=self.path,
            name=self.name,
            split=self.split,
            trust_remote_code=self.trust_remote_code,
        )
        return dataset
