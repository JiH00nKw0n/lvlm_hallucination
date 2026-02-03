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
    require_image: bool = True
    num_sample: int = 10_000
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
                if image is not None:
                    return True
                images = example.get("images", None)
                if images is None:
                    return False
                if isinstance(images, list):
                    return len(images) > 0 and images[0] is not None
                return images is not None

            dataset = dataset.filter(_has_image)

        if self.num_sample is not None and len(dataset) > self.num_sample:
            dataset = (
                dataset
                .shuffle(seed=self.seed)
                .select(range(self.num_sample))
            )

        return dataset
