from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder("COCOKarpathyDatasetBuilder")
class COCOKarpathyDatasetBuilder(BaseBuilder):
    path: str = "namkha1032/coco-karpathy"
    split: Optional[str] = "train"
    name: Optional[str] = None
    trust_remote_code: bool = False
    require_image: bool = True
    num_sample: Optional[int] = None
    seed: int = 42

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            path=self.path,
            name=self.name,
            split=self.split,
            trust_remote_code=self.trust_remote_code,
        )

        if self.require_image:
            def _has_image(example) -> bool:
                image = example.get("image", None)
                return image is not None

            dataset = dataset.filter(_has_image)

        if self.num_sample is not None and len(dataset) > self.num_sample:
            dataset = (
                dataset
                .shuffle(seed=self.seed)
                .select(range(self.num_sample))
            )

        return dataset
